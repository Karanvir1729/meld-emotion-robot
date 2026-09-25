"""Evaluate finished runs on the MELD test split and write runs/report.md + runs/metrics.json."""

import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from emo.config import CERTAINTY_THRESHOLDS, DEPLOYED_RUN, EMOTIONS, RUNS_DIR
from emo.data import load_manifest
from emo.model import primary_head

DISPLAY_NAMES = {
    "text": "text only",
    "audio": "audio only",
    "vision": "vision only",
    "text-vision": "text + vision",
    "text-vision-nocontext": "text + vision, no context",
    "text-audio": "text + audio",
    "text-audio-vision": "text + vision + audio",
}
VIEW_LABEL = {"text": "the words", "audio": "the voice", "vision": "the face"}


def weighted_f1(labels, preds) -> float:
    return float(f1_score(labels, preds, average="weighted"))


def main() -> None:
    runs = {name: load_run(name) for name in DISPLAY_NAMES if (RUNS_DIR / name / "logits_test.pt").exists()}
    if not runs:
        raise SystemExit("no finished runs under runs/; train first")
    first = next(iter(runs.values()))
    assert all(run["ids"] == first["ids"] for run in runs.values()), "runs were scored on different test manifests"
    labels = first["labels"]
    manifest = {row["id"]: row for row in load_manifest("test")}
    present = {m: np.array([m == "text" or manifest[i][f"has_{m}"] for i in first["ids"]]) for m in ("text", "audio", "vision")}  # as live
    metrics = {"n_test": int(len(labels)), "deployed": DEPLOYED_RUN}
    report = ["# Results on the MELD test split", ""]

    # 1. Classification table: every run's primary head, plus the deployed model's single-modality heads.
    rows = [("majority class (neutral)", scores(labels, np.zeros_like(labels)))]
    rows += [(display(name), scores(labels, run["preds"])) for name, run in runs.items()]
    deployed = runs.get(DEPLOYED_RUN)
    if deployed:
        rows += [(f"deployed model, {head} head only (views.{head}; {present[head].sum()} turns with that input)", scores(labels[present[head]], deployed["heads"][head][present[head]]))
                 for head in deployed["info"]["modalities"]]
    metrics["classification"] = dict(rows)
    report += ["## Classification", "", table(["model", "weighted-F1", "macro-F1", "accuracy"] + EMOTIONS, [
        [name, s["weighted_f1"], s["macro_f1"], s["accuracy"]] + [s["per_class_f1"][e] for e in EMOTIONS] for name, s in rows
    ]), ""]

    # 2. Does another modality add measurable value over text alone? Paired bootstrap over test utterances.
    if "text" in runs:
        text_preds, gains = runs["text"]["preds"], {}
        for name, run in runs.items():
            if name != "text" and "text" in run["info"]["modalities"]:
                low, high = bootstrap_delta(labels, run["preds"], text_preds)
                delta = weighted_f1(labels, run["preds"]) - weighted_f1(labels, text_preds)
                gains[display(name)] = {"delta_weighted_f1": round(delta, 4), "ci95": [round(low, 4), round(high, 4)]}
        metrics["gain_over_text"] = gains
        report += ["## Gain over text only (weighted-F1, paired bootstrap over test utterances, 1000 resamples)", "",
                   table(["model", "difference", "95% interval"], [[n, f"{g['delta_weighted_f1']:+.3f}", f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]"] for n, g in gains.items()]), ""]

    if deployed and len(deployed["info"]["modalities"]) > 1:
        # 3. When the views disagree, who is right? Only views whose input is present count, as live.
        views = {m: deployed["heads"][m] for m in deployed["info"]["modalities"]}
        shown = np.stack([np.where(present[m], v, -1) for m, v in views.items()])  # -1: no input, no view
        disagree = np.array([len(set(column[column >= 0])) > 1 for column in shown.T])
        subset = {"share_of_test": float(disagree.mean()), "accuracy_fused": float(accuracy_score(labels[disagree], deployed["preds"][disagree]))}
        subset.update({f"accuracy_{m}": float(accuracy_score(labels[disagree & present[m]], v[disagree & present[m]])) for m, v in views.items()})
        metrics["disagreement"] = subset
        report += ["## Turns where the views disagree", "",
                   f"{subset['share_of_test']:.1%} of test turns. Accuracy on them: fused {subset['accuracy_fused']:.3f}, "
                   + ", ".join(f"{VIEW_LABEL[m]} {subset[f'accuracy_{m}']:.3f}" for m in views), ""]

        # 4. Which WavLM layers an audio branch learned to use.
        if "audio" in deployed["info"]["modalities"]:
            state = torch.load(RUNS_DIR / DEPLOYED_RUN / "model.pt", map_location="cpu")
            layer_mix = torch.softmax(state["layer_weights"].float(), dim=0).tolist()
            metrics["wavlm_layer_weights"] = [round(w, 3) for w in layer_mix]
            report += ["## Learned WavLM layer weights (0 = CNN output, 12 = last transformer layer)", "",
                       table(["layer"] + list(range(13)), [["weight"] + layer_mix]), ""]

    if deployed:
        # 5. Are the certainty buckets meaningful? Accuracy per bucket after temperature scaling.
        confidence = deployed["probs"].max(1)
        buckets = []
        for name, lo, hi in (("high", CERTAINTY_THRESHOLDS["high"], 1.01), ("medium", CERTAINTY_THRESHOLDS["medium"], CERTAINTY_THRESHOLDS["high"]), ("low", 0.0, CERTAINTY_THRESHOLDS["medium"])):
            mask = (confidence >= lo) & (confidence < hi)
            buckets.append({"certainty": name, "share": float(mask.mean()), "accuracy": float(accuracy_score(labels[mask], deployed["preds"][mask])) if mask.any() else None})
        metrics["calibration"] = {"temperature": deployed["info"]["temperature"], "buckets": buckets}
        report += [f"## Certainty buckets (temperature {deployed['info']['temperature']})", "",
                   table(["certainty", "share of test", "accuracy"], [[b["certainty"], b["share"], b["accuracy"]] for b in buckets]), ""]

        # 6. Confusion matrix of the deployed model.
        cm = confusion_matrix(labels, deployed["preds"], labels=range(len(EMOTIONS)))
        metrics["confusion_matrix"] = cm.tolist()
        report += ["## Confusion matrix (rows = true, columns = predicted)", "",
                   table(["true \\ predicted"] + EMOTIONS, [[e] + row for e, row in zip(EMOTIONS, cm.tolist())]), ""]

    (RUNS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RUNS_DIR / "report.md").write_text("\n".join(report))
    print("\n".join(report))


def display(name: str) -> str:
    return DISPLAY_NAMES[name] + (" (deployed)" if name == DEPLOYED_RUN else "")


def load_run(name: str) -> dict:
    """Test labels, argmax of every head, and calibrated probabilities of the primary head."""
    data = torch.load(RUNS_DIR / name / "logits_test.pt")
    info = json.loads((RUNS_DIR / name / "run.json").read_text())
    probs = torch.softmax(data["logits"][primary_head(tuple(info["modalities"]))] / info["temperature"], dim=-1).numpy()
    return {"ids": data["ids"], "labels": data["labels"].numpy(), "heads": {h: l.argmax(1).numpy() for h, l in data["logits"].items()}, "probs": probs, "preds": probs.argmax(1), "info": info}


def scores(labels: np.ndarray, preds: np.ndarray) -> dict:
    per_class = f1_score(labels, preds, average=None, labels=range(len(EMOTIONS)), zero_division=0)
    return {
        "weighted_f1": weighted_f1(labels, preds),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "per_class_f1": dict(zip(EMOTIONS, per_class.tolist())),
    }


def bootstrap_delta(labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of weighted-F1(a) - weighted-F1(b) over paired resamples of the test utterances."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        deltas.append(weighted_f1(labels[idx], preds_a[idx]) - weighted_f1(labels[idx], preds_b[idx]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def table(headers: list, rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


if __name__ == "__main__":
    main()
