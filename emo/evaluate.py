"""Evaluate finished runs on the MELD test split and write runs/report.md + runs/metrics.json."""

import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from emo.config import CERTAINTY_THRESHOLDS, DEPLOYED_RUN, EMOTIONS, RUNS_DIR

RUN_ORDER = ["text", "audio", "both-nocontext", "both"]


def weighted_f1(labels, preds) -> float:
    return float(f1_score(labels, preds, average="weighted"))


def main() -> None:
    runs = {name: load_run(name) for name in RUN_ORDER if (RUNS_DIR / name / "logits_test.pt").exists()}
    if not runs:
        raise SystemExit("no finished runs under runs/; train first")
    labels = next(iter(runs.values()))["labels"]
    metrics = {"n_test": int(len(labels))}
    report = ["# Results on the MELD test split", ""]

    # 1. Classification table: every run's primary head, plus the deployed model's single-modality heads.
    rows = [("majority class (neutral)", scores(labels, np.zeros_like(labels)))]
    rows += [(name, scores(labels, run["preds"])) for name, run in runs.items()]
    if DEPLOYED_RUN in runs:
        rows += [(f"{DEPLOYED_RUN}: {head} head only", scores(labels, runs[DEPLOYED_RUN]["heads"][head])) for head in ("text", "audio")]
    metrics["classification"] = dict(rows)
    report += ["## Classification", "", table(["model", "weighted-F1", "macro-F1", "accuracy"] + EMOTIONS, [
        [name, s["weighted_f1"], s["macro_f1"], s["accuracy"]] + [s["per_class_f1"][e] for e in EMOTIONS] for name, s in rows
    ]), ""]

    # 2. Does audio add measurable value? Paired bootstrap over test utterances.
    if {"both", "text"} <= runs.keys():
        low, high = bootstrap_delta(labels, runs["both"]["preds"], runs["text"]["preds"])
        delta = rows_lookup(rows, "both")["weighted_f1"] - rows_lookup(rows, "text")["weighted_f1"]
        metrics["audio_gain"] = {"delta_weighted_f1": round(delta, 4), "ci95": [round(low, 4), round(high, 4)]}
        report += ["## Audio gain over text (paired bootstrap, 1000 resamples)", "",
                   f"weighted-F1(both) - weighted-F1(text) = {delta:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]", ""]

    if DEPLOYED_RUN in runs:
        run = runs[DEPLOYED_RUN]
        # 3. When the words and the voice disagree, who is right?
        text_pred, audio_pred = run["heads"]["text"], run["heads"]["audio"]
        disagree = text_pred != audio_pred
        subset = {"share_of_test": float(disagree.mean())}
        subset.update({f"accuracy_{h}": float(accuracy_score(labels[disagree], p[disagree])) for h, p in (("fused", run["preds"]), ("text", text_pred), ("audio", audio_pred))})
        metrics["disagreement"] = subset
        report += ["## Turns where the text head and the audio head disagree", "",
                   f"{subset['share_of_test']:.1%} of test turns. Accuracy on them: fused {subset['accuracy_fused']:.3f}, text head {subset['accuracy_text']:.3f}, audio head {subset['accuracy_audio']:.3f}", ""]

        # 4. Which WavLM layers the audio branch learned to use.
        state = torch.load(RUNS_DIR / DEPLOYED_RUN / "model.pt", map_location="cpu")
        layer_mix = torch.softmax(state["layer_weights"].float(), dim=0).tolist()
        metrics["wavlm_layer_weights"] = [round(w, 3) for w in layer_mix]
        report += ["## Learned WavLM layer weights (0 = CNN output, 12 = last transformer layer)", "",
                   table(["layer"] + list(range(13)), [["weight"] + layer_mix]), ""]

        # 5. Are the certainty buckets meaningful? Accuracy per bucket after temperature scaling.
        confidence = run["probs"].max(1)
        buckets = []
        for name, lo, hi in (("high", CERTAINTY_THRESHOLDS["high"], 1.01), ("medium", CERTAINTY_THRESHOLDS["medium"], CERTAINTY_THRESHOLDS["high"]), ("low", 0.0, CERTAINTY_THRESHOLDS["medium"])):
            mask = (confidence >= lo) & (confidence < hi)
            buckets.append({"certainty": name, "share": float(mask.mean()), "accuracy": float(accuracy_score(labels[mask], run["preds"][mask])) if mask.any() else None})
        metrics["calibration"] = {"temperature": run["info"]["temperature"], "buckets": buckets}
        report += [f"## Certainty buckets (temperature {run['info']['temperature']})", "",
                   table(["certainty", "share of test", "accuracy"], [[b["certainty"], b["share"], b["accuracy"]] for b in buckets]), ""]

        # 6. Confusion matrix of the deployed model.
        cm = confusion_matrix(labels, run["preds"], labels=range(len(EMOTIONS)))
        metrics["confusion_matrix"] = cm.tolist()
        report += ["## Confusion matrix (rows = true, columns = predicted)", "",
                   table(["true \\ predicted"] + EMOTIONS, [[e] + row for e, row in zip(EMOTIONS, cm.tolist())]), ""]

    (RUNS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RUNS_DIR / "report.md").write_text("\n".join(report))
    print("\n".join(report))


def load_run(name: str) -> dict:
    """Test labels, argmax of every head, and calibrated probabilities of the primary head."""
    data = torch.load(RUNS_DIR / name / "logits_test.pt")
    info = json.loads((RUNS_DIR / name / "run.json").read_text())
    primary = "fused" if info["modalities"] == "both" else info["modalities"]
    probs = torch.softmax(data["logits"][primary] / info["temperature"], dim=-1).numpy()
    return {"labels": data["labels"].numpy(), "heads": {h: l.argmax(1).numpy() for h, l in data["logits"].items()}, "probs": probs, "preds": probs.argmax(1), "info": info}


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


def rows_lookup(rows: list[tuple[str, dict]], name: str) -> dict:
    return dict(rows)[name]


def table(headers: list, rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


if __name__ == "__main__":
    main()
