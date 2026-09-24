"""Latency, memory and parameter budget of the deployed pipeline on this machine -> runs/bench_<device>.json."""

import argparse
import json
import platform
import resource
import subprocess
import time

import numpy as np
import torch
import transformers

from emo.config import DEPLOYED_RUN, REALTIME_TARGETS_MS, RUNS_DIR, pick_device
from emo.data import load_manifest
from emo.responder import Responder
from emo.session import Classifier, Session, parameter_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    parser.add_argument("--turns", type=int, default=200, help="measured test utterances, in dialogue order")
    parser.add_argument("--warmup", type=int, default=10, help="turns run before measuring")
    args = parser.parse_args()
    device = pick_device(args.device)

    start = time.perf_counter()
    classifier, responder = Classifier(DEPLOYED_RUN, device), Responder(device)
    load_s = time.perf_counter() - start
    session = Session(classifier, responder)
    rows = [row for row in load_manifest("test") if row["has_audio"]][: args.warmup + args.turns]

    turns = []
    for i, row in enumerate(rows):
        events = list(session.step(row["text"], row["audio_path"]))
        if i == 0:
            cold_start_s = time.perf_counter() - start
        if i < args.warmup:
            continue
        state, done = events[0], events[-1]
        turns.append({**done["latency_ms"], "audio_s": state["audio_seconds"], "tokens": sum(e["event"] == "token" for e in events), "source": done["source"]})

    p = lambda key, q: float(np.percentile([t[key] for t in turns], q))
    decode_s = sum((t["response"] - t["first_token"]) / 1000 for t in turns)
    result = {
        "hardware": hardware(),
        "device": str(device),
        "turns_measured": len(turns),
        "model_load_s": round(load_s, 1),
        "cold_start_s": round(cold_start_s, 1),
        "latency_ms": {key: {"p50": round(p(key, 50)), "p95": round(p(key, 95)), "target_p95": REALTIME_TARGETS_MS[key]} for key in REALTIME_TARGETS_MS},
        "decode_tokens_per_s": round(sum(t["tokens"] for t in turns) / decode_s, 1),
        "audio_real_time_factor": round(float(np.median([t["state"] / 1000 / t["audio_s"] for t in turns])), 3),
        "fallback_rate": round(sum(t["source"] == "fallback" for t in turns) / len(turns), 3),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "mps_allocated_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if device.type == "mps" else None,
        "parameters_millions": parameter_counts(classifier, responder),
    }
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"bench_{device.type}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def hardware() -> dict:
    sysctl = lambda key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(sysctl("hw.memsize")) / 1e9),
        "os": f"macOS {subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True).stdout.strip()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


if __name__ == "__main__":
    main()
