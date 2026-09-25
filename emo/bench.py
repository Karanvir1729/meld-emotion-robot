"""Latency, memory and parameter budget of the deployed pipeline on this machine -> runs/bench_<device>.json."""

import argparse
import json
import os
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
import transformers

from emo.config import DEPLOYED_RUN, REALTIME_TARGETS_MS, RUNS_DIR, pick_device
from emo.data import load_manifest, visual_input
from emo.responder import load_responder
from emo.session import Classifier, Session, parameter_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEPLOYED_RUN)
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    parser.add_argument("--turns", type=int, default=200, help="measured test utterances, in dialogue order")
    parser.add_argument("--warmup", type=int, default=10, help="turns run before measuring")
    args = parser.parse_args()
    device = pick_device(args.device)

    start = time.perf_counter()
    classifier, responder = Classifier(args.run, device), load_responder(device)
    load_s = time.perf_counter() - start
    session = Session(classifier, responder)
    needed = [m for m in classifier.model.modalities if m != "text"]
    rows = [row for row in load_manifest("test") if all(row[f"has_{m}"] for m in needed)][: args.warmup + args.turns]
    live_video = sum(Path(r["video_path"]).exists() for r in rows[args.warmup:]) if classifier.face_encoder else 0  # turns that decode a real clip

    turns = []
    for i, row in enumerate(rows):
        events = list(session.step(row["text"], row["audio_path"], visual_input(row)))
        if i == 0:
            cold_start_s = time.perf_counter() - start
        if i < args.warmup:
            continue
        done = events[-1]
        latency = {**done["latency_ms"], "first_token": done["latency_ms"]["first_token"] or done["latency_ms"]["response"]}  # empty reply: no token
        generated = "".join(e["text"] for e in events if e["event"] == "token")
        turns.append({**latency, "tokens": len(responder.tokenizer.encode(generated, add_special_tokens=False)), "source": done["source"]})

    def percentile(key: str, q: int) -> int:
        return round(float(np.percentile([t[key] for t in turns], q)))

    decode_s = sum((t["response"] - t["first_token"]) / 1000 for t in turns)
    result = {
        "hardware": hardware(),
        "device": str(device),
        "turns_measured": len(turns),
        "turns_from_raw_video": live_video,
        "model_load_s": round(load_s, 1),
        "cold_start_s": round(cold_start_s, 1),
        "latency_ms": {key: {"p50": percentile(key, 50), "p95": percentile(key, 95), "target_p95": REALTIME_TARGETS_MS[key]} for key in REALTIME_TARGETS_MS},
        "decode_tokens_per_s": round(sum(t["tokens"] for t in turns) / decode_s, 1),
        "fallback_rate": round(sum(t["source"] == "fallback" for t in turns) / len(turns), 3),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 1e9, 2),
        "mps_allocated_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if device.type == "mps" else None,
        "mlx_peak_gb": mlx_peak_gb(),
        "parameters_millions": parameter_counts(classifier, responder),
    }
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"bench_{device.type}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def hardware() -> dict:
    """CPU name, memory and versions; the CPU name needs a platform-specific call."""
    cpu = shell("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30),
        "os": f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def mlx_peak_gb() -> float | None:
    """Peak memory of the MLX runtime (a 4-bit reply model lives there, outside torch's accounting)."""
    if "mlx.core" not in sys.modules:
        return None
    import mlx.core as mx

    return round(mx.get_peak_memory() / 1e9, 2)


def shell(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
