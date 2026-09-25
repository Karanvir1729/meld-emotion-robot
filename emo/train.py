"""Train an EmotionModel on MELD; saves the checkpoint, calibrated logits and a summary under runs/<name>/."""

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from emo import audio, config, vision
from emo.data import Collate, MeldDataset, load_manifest
from emo.evaluate import weighted_f1
from emo.model import EmotionModel, canonical, count_parameters, save

FEATURE_LOADERS = {"audio": audio.load_features, "vision": vision.load_features}
MODEL_INPUTS = ("tokens", "audio", "audio_present", "vision", "vision_present")


def main() -> None:
    args = parse_args()
    modalities = canonical(args.modalities.split(","))
    name = args.name or run_name(modalities, args.context)
    run_dir = config.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = config.pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL) if "text" in modalities else None
    loaders = {split: make_loader(split, modalities, args, tokenizer) for split in config.SPLITS}
    model = EmotionModel(modalities).to(device)
    epochs = args.epochs or (config.EPOCHS_WITH_TEXT if "text" in modalities else config.EPOCHS_HEADS_ONLY)
    total_steps = epochs * len(loaders["train"])
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_FRACTION * total_steps), total_steps)
    print(f"{name}: {count_parameters(model) / 1e6:.1f} M parameters, {len(loaders['train'].dataset)} training utterances, {epochs} epochs on {device}")

    best_f1, best_epoch, best_state, start = -1.0, 0, None, time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, device)
        labels, logits = predict(model, loaders["dev"], device)
        dev_f1 = weighted_f1(labels, logits[model.primary].argmax(1))
        print(f"epoch {epoch}: train loss {loss:.3f}, dev weighted-F1 {dev_f1:.4f}, {(time.perf_counter() - start) / 60:.1f} min", flush=True)
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    view_f1 = fit_view_heads(model, loaders, device) if len(modalities) > 1 else {}

    dev_labels, dev_logits = predict(model, loaders["dev"], device)
    model.temperature.fill_(fit_temperature(dev_logits[model.primary], dev_labels))
    test_labels, test_logits = predict(model, loaders["test"], device)
    for split, labels, logits in (("dev", dev_labels, dev_logits), ("test", test_labels, test_logits)):
        ids = [row["id"] for row in loaders[split].dataset.rows]
        torch.save({"ids": ids, "labels": labels, "logits": logits}, run_dir / f"logits_{split}.pt")
    save(model, run_dir / "model.pt")

    summary = {
        **vars(args),
        "modalities": list(modalities),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "dev_weighted_f1": round(best_f1, 4),
        "view_heads_dev_weighted_f1": view_f1,
        "temperature": round(model.temperature.item(), 3),
        "train_minutes": round((time.perf_counter() - start) / 60, 1),
        "parameters": count_parameters(model),
        "device": str(device),
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modalities", default="text,vision", help="comma-separated subset of text,audio,vision")
    parser.add_argument("--context", type=int, choices=[0, 1], default=1, help="feed the previous utterance to the text encoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="override the default for the modalities")
    parser.add_argument("--limit", type=int, help="use only the first N utterances of every split (smoke test)")
    parser.add_argument("--name", help="run directory name (default: derived from the flags)")
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    return parser.parse_args()


def run_name(modalities: tuple[str, ...], context: int) -> str:
    return "-".join(modalities) + ("" if context else "-nocontext")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(split: str, modalities: tuple[str, ...], args: argparse.Namespace, tokenizer) -> DataLoader:
    rows = load_manifest(split)[: args.limit]
    features = {m: FEATURE_LOADERS[m](split) for m in modalities if m != "text"}
    return DataLoader(MeldDataset(rows, features), batch_size=config.BATCH_SIZE, shuffle=split == "train", collate_fn=Collate(tokenizer, bool(args.context)))


def param_groups(model: EmotionModel) -> list[dict]:
    """A small learning rate for the pretrained text encoder, a larger one for everything trained from scratch."""
    encoder = [p for n, p in model.named_parameters() if n.startswith("text_encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("text_encoder.")]
    groups = [{"params": encoder, "lr": config.LR_ENCODER}, {"params": rest, "lr": config.LR_HEAD}]
    return [g for g in groups if g["params"]]


def train_one_epoch(model: EmotionModel, loader: DataLoader, optimizer, scheduler, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        if len(model.modalities) > 1:  # modality dropout: hide some audio / faces so the fused head also learns to work without them
            for modality in ("audio", "vision"):
                if f"{modality}_present" in batch:
                    batch[f"{modality}_present"] &= torch.rand(len(batch["labels"]), device=device) > config.MODALITY_DROPOUT
        logits = model(**model_inputs(batch))
        loss = compute_loss(model, logits, batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def compute_loss(model: EmotionModel, logits: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    """Cross-entropy on the primary head; a fused model also trains every branch head, weighted down."""
    labels = batch["labels"]
    loss = F.cross_entropy(logits[model.primary], labels)
    if model.primary == "fused":
        for modality in model.modalities:
            present = batch.get(f"{modality}_present")  # text is always present
            if present is None:
                loss = loss + config.AUX_LOSS_WEIGHT * F.cross_entropy(logits[modality], labels)
            elif present.any():
                loss = loss + config.AUX_LOSS_WEIGHT * F.cross_entropy(logits[modality][present], labels[present])
    return loss


def fit_view_heads(model: EmotionModel, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, float]:
    """Refit each modality's own head on frozen branch vectors; returns each head's best dev weighted-F1.

    In the joint run a view head gets ~3 epochs of a decaying learning rate while the fused head does the work,
    and the vision head collapsed to "neutral". Everything else is frozen here, so the model's prediction does
    not change; only the `views` do. Each head is scored on the utterances where its input is present.
    """
    (train_x, train_present, train_y), (dev_x, dev_present, dev_y) = (branch_vectors(model, loaders[s], device) for s in ("train", "dev"))
    scores = {}
    for modality in model.modalities:
        head = getattr(model, f"{modality}_head")
        x, y = train_x[modality][train_present[modality]], train_y[train_present[modality]]
        dev_mask = dev_present[modality]
        optimizer = torch.optim.AdamW(head.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
        best_f1, best_state = -1.0, None
        for _ in range(config.EPOCHS_HEADS_ONLY):
            for idx in torch.randperm(len(y), device=device).split(config.BATCH_SIZE):
                loss = F.cross_entropy(head(x[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                f1 = weighted_f1(dev_y[dev_mask].cpu(), head(dev_x[modality][dev_mask]).argmax(1).cpu())
            if f1 > best_f1:
                best_f1, best_state = f1, {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        scores[modality] = round(best_f1, 4)
    return scores


@torch.no_grad()
def branch_vectors(model: EmotionModel, loader: DataLoader, device: torch.device) -> tuple[dict, dict, torch.Tensor]:
    """Frozen branch vectors, input-presence masks and labels for a whole split, on the device."""
    model.eval()
    vectors, present, labels = defaultdict(list), defaultdict(list), []
    for batch in loader:
        batch = to_device(batch, device)
        for modality, vector in model.branch_vectors(**model_inputs(batch)).items():
            vectors[modality].append(vector)
            present[modality].append(batch.get(f"{modality}_present", torch.ones(len(vector), dtype=torch.bool, device=device)))
        labels.append(batch["labels"])
    return {m: torch.cat(v) for m, v in vectors.items()}, {m: torch.cat(p) for m, p in present.items()}, torch.cat(labels)


@torch.no_grad()
def predict(model: EmotionModel, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Labels and raw logits of every head for a whole split, as CPU tensors."""
    model.eval()
    labels, logits = [], defaultdict(list)
    for batch in loader:
        batch = to_device(batch, device)
        for head, values in model(**model_inputs(batch)).items():
            logits[head].append(values.cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(labels), {head: torch.cat(values) for head, values in logits.items()}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """One scalar that minimises dev negative log-likelihood (temperature scaling, Guo et al. 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def model_inputs(batch: dict) -> dict:
    return {k: batch[k] for k in MODEL_INPUTS if k in batch}


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    main()
