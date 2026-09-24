"""Dev-only checks of the reply generator: fallback rate, length, and whether replies follow the emotion state."""

import argparse
import json
from collections import Counter

from emo.config import DEPLOYED_RUN, EMOTIONS, RUNS_DIR, pick_device
from emo.data import load_manifest
from emo.responder import Responder, finish, render_messages
from emo.session import Classifier, Session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-emotion", type=int, default=15, help="dev utterances per true emotion")
    parser.add_argument("--run", default=DEPLOYED_RUN)
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    args = parser.parse_args()
    device = pick_device(args.device)
    classifier, responder = Classifier(args.run, device), Responder(device)

    counts, rows = Counter(), []
    for row in load_manifest("dev"):
        if row["has_audio"] and counts[row["emotion"]] < args.per_emotion:
            counts[row["emotion"]] += 1
            rows.append(row)

    records = []
    for row in rows:
        events = list(Session(classifier, responder).step(row["text"], row["audio_path"]))  # fresh memory: single turn
        state, done = events[0], events[-1]
        # Counterfactual: the same turn with only the emotion fields changed, greedy decoding on both sides.
        other = EMOTIONS[(EMOTIONS.index(state["emotion"]) + 1) % len(EMOTIONS)]
        altered = {**state, "emotion": other, "views": {"text": other, "audio": other, "agree": True}}
        greedy = finish(state["emotion"], "".join(responder.stream(render_messages(state), sample=False)))[0]
        counterfactual = finish(other, "".join(responder.stream(render_messages(altered), sample=False)))[0]
        records.append({"text": row["text"], "meld_label": row["emotion"], "state": state["emotion"], "reply": done["response"], "source": done["source"],
                        "greedy_reply": greedy, "counterfactual_emotion": other, "counterfactual_reply": counterfactual, "changed": greedy != counterfactual})

    summary = {
        "utterances": len(records),
        "fallback_rate": round(sum(r["source"] == "fallback" for r in records) / len(records), 3),
        "mean_words": round(sum(len(r["reply"].split()) for r in records) / len(records), 1),
        "changed_by_counterfactual_emotion": round(sum(r["changed"] for r in records) / len(records), 3),
    }
    (RUNS_DIR / "responses.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    lines = ["# Reply checks on dev utterances", "", json.dumps(summary), "", "| person said | state | Pip (sampled) | if the state were... | Pip would say (greedy) |", "|---|---|---|---|---|"]
    seen = set()
    for r in records:  # one example per predicted emotion
        if r["state"] not in seen:
            seen.add(r["state"])
            lines.append(f"| {r['text']} | {r['state']} | {r['reply']} | {r['counterfactual_emotion']} | {r['counterfactual_reply']} |")
    (RUNS_DIR / "responses.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
