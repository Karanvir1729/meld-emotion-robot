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

from emo import config
from emo.audio import load_features
from emo.data import Collate, MeldDataset, load_manifest
from emo.evaluate import weighted_f1
from emo.model import EmotionModel, count_parameters, save


def main() -> None:
    args = parse_args()
    name = args.name or args.modalities + ("" if args.context else "-nocontext")
    run_dir = config.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = config.pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL) if args.modalities != "audio" else None
    loaders = {split: make_loader(split, args, tokenizer) for split in config.SPLITS}
    model = EmotionModel(args.modalities).to(device)
    epochs = args.epochs or config.EPOCHS[args.modalities]
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

    dev_labels, dev_logits = predict(model, loaders["dev"], device)
    model.temperature.fill_(fit_temperature(dev_logits[model.primary], dev_labels))
    test_labels, test_logits = predict(model, loaders["test"], device)
    for split, labels, logits in (("dev", dev_labels, dev_logits), ("test", test_labels, test_logits)):
        ids = [row["id"] for row in loaders[split].dataset.rows]
        torch.save({"ids": ids, "labels": labels, "logits": logits}, run_dir / f"logits_{split}.pt")
    save(model, run_dir / "model.pt")

    summary = {
        **vars(args),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "dev_weighted_f1": round(best_f1, 4),
        "temperature": round(model.temperature.item(), 3),
        "train_minutes": round((time.perf_counter() - start) / 60, 1),
        "parameters": count_parameters(model),
        "device": str(device),
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modalities", choices=["text", "audio", "both"], default="both")
    parser.add_argument("--context", type=int, choices=[0, 1], default=1, help="feed the previous utterance to the text encoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="override the default for the modality")
    parser.add_argument("--limit", type=int, help="use only the first N utterances of every split (smoke test)")
    parser.add_argument("--name", help="run directory name (default: derived from the flags)")
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(split: str, args: argparse.Namespace, tokenizer) -> DataLoader:
    rows = load_manifest(split)[: args.limit]
    features = load_features(split) if args.modalities != "text" else None
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
        if model.modalities == "both":  # modality dropout: hide some audio so the fused head also learns to work without it
            batch["audio_present"] &= torch.rand(len(batch["labels"]), device=device) > config.MODALITY_DROPOUT
        logits = model(batch.get("tokens"), batch["audio"], batch["audio_present"])
        loss = compute_loss(model, logits, batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def compute_loss(model: EmotionModel, logits: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    """Cross-entropy on the primary head; a "both" model also trains its text and audio heads, weighted down."""
    labels = batch["labels"]
    loss = F.cross_entropy(logits[model.primary], labels)
    if model.modalities == "both":
        present = batch["audio_present"]
        loss = loss + config.AUX_LOSS_WEIGHT * F.cross_entropy(logits["text"], labels)
        if present.any():
            loss = loss + config.AUX_LOSS_WEIGHT * F.cross_entropy(logits["audio"][present], labels[present])
    return loss


@torch.no_grad()
def predict(model: EmotionModel, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Labels and raw logits of every head for a whole split, as CPU tensors."""
    model.eval()
    labels, logits = [], defaultdict(list)
    for batch in loader:
        batch = to_device(batch, device)
        for head, values in model(batch.get("tokens"), batch["audio"], batch["audio_present"]).items():
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


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    main()
