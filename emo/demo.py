"""Replay a MELD dialogue, or one utterance of your own (text + video or image), through the whole pipeline and print both outputs."""

import argparse
import json

from emo.config import DEPLOYED_RUN, pick_device
from emo.data import load_manifest, visual_input
from emo.responder import load_responder
from emo.session import Classifier, Session


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    session = Session(Classifier(args.run, device), load_responder(device))
    if args.text:
        turns = [{"speaker": "person", "text": args.text, "audio_path": args.audio, "video": args.video}]
    else:
        turns = [{**row, "video": visual_input(row), "audio_path": row["audio_path"] if row["has_audio"] else None}
                 for row in load_manifest(args.split) if row["dialogue_id"] == args.dialogue]
    log = open(args.jsonl, "w") if args.jsonl else None
    for row in turns:
        for event in session.step(row["text"], row["audio_path"], row["video"]):
            show(event, row)
            if log:
                log.write(json.dumps(event) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialogue", type=int, default=0, help="MELD dialogue id to replay")
    parser.add_argument("--text", help="your own utterance (skips the MELD replay)")
    parser.add_argument("--audio", help="16 kHz mono WAV/FLAC of that utterance (optional)")
    parser.add_argument("--video", help="video or image of the person saying it, or a directory of face crops (optional)")
    parser.add_argument("--run", default=DEPLOYED_RUN)
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    parser.add_argument("--jsonl", help="also write every event to this file")
    args = parser.parse_args()
    if (args.audio or args.video) and not args.text:
        parser.error("--audio / --video need --text: the transcript is an input, there is no speech recognition")
    return args


def show(event: dict, row: dict) -> None:
    if event["event"] == "state":
        views = event["views"]
        label = f"  MELD label: {row['emotion']}" if "emotion" in row else ""
        print(f"\n[{event['turn']}] {row['speaker']}: {event['text']}")
        opinions = "  ".join(f"{name}={views[m]}" for m, name in (("text", "words"), ("vision", "face"), ("audio", "voice")) if m in views)
        print(f"    state: {event['emotion']} ({event['confidence']:.2f}, {event['certainty']})  {opinions}  agree={views['agree']}{label}  [{event['latency_ms']['state']} ms]")
        print("    Pip: ", end="", flush=True)
    elif event["event"] == "token":
        print(event["text"], end="", flush=True)
    else:
        if event["source"] == "fallback":
            print(f"\n    Pip (fallback): {event['response']}", end="")
        print(f"\n    [first token {event['latency_ms']['first_token']} ms, done {event['latency_ms']['response']} ms]")


if __name__ == "__main__":
    main()
