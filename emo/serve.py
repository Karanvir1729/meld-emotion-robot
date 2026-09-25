"""JSON Lines over stdin/stdout: one request per line in, the turn's events streamed back one per line.

Request:  {"session_id": "s1", "text": "I said I was fine.", "video_path": "clip.mp4", "audio_path": "clip.flac"}   (video / audio optional)
          {"session_id": "s1", "reset": true}
Events:   ready, state, token, done, reset, error (see README for the fields).
"""

import argparse
import json
import sys

from emo.config import DEPLOYED_RUN, pick_device
from emo.responder import load_responder
from emo.session import Classifier, Session, parameter_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEPLOYED_RUN)
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    args = parser.parse_args()
    device = pick_device(args.device)
    classifier, responder = Classifier(args.run, device), load_responder(device)
    sessions: dict[str, Session] = {}
    emit({"event": "ready", "device": str(device), "parameters_millions": parameter_counts(classifier, responder)})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            session_id = request.get("session_id", "default")
            if request.get("reset"):
                sessions.pop(session_id, None)
                emit({"event": "reset", "session_id": session_id})
                continue
            session = sessions.setdefault(session_id, Session(classifier, responder, session_id))
            for event in session.step(request["text"], request.get("audio_path"), request.get("video_path")):
                emit(event)
        except Exception as error:  # a bad request must not take the server down
            emit({"event": "error", "message": f"{type(error).__name__}: {error}",
                  "hint": 'send {"session_id", "text", "video_path" (video/image, optional), "audio_path" (16 kHz mono WAV/FLAC, optional)} or {"session_id", "reset": true}'})


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
