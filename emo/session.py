"""One conversation with the robot: classify each utterance, emit a state event, then stream a grounded reply."""

import json
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import AutoTokenizer

from emo.audio import AudioEncoder, load_audio
from emo.config import (
    AUDIO_DIM,
    AUDIO_LAYERS,
    CERTAINTY_THRESHOLDS,
    DEPLOYED_RUN,
    EMOTIONS,
    FACE_DETECTOR_PARAMS,
    MEMORY_MESSAGES,
    RUNS_DIR,
    SAMPLE_RATE,
    TEXT_MODEL,
    VISION_DIM,
    pick_device,
)
from emo.data import tokenize
from emo.model import count_parameters, load
from emo.responder import FALLBACK, finish, render_messages
from emo.faces import faces_from_video, load_faces
from emo.vision import FaceEncoder


class Classifier:
    """A trained run, ready for one utterance at a time: (text, previous text, audio, faces) -> emotion fields."""

    def __init__(self, run: str = DEPLOYED_RUN, device: torch.device | None = None):
        self.device = device or pick_device()
        info = json.loads((RUNS_DIR / run / "run.json").read_text())
        self.context = bool(info["context"])
        self.model = load(RUNS_DIR / run / "model.pt", info["modalities"], self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL) if self.model.use_text else None
        self.audio_encoder = AudioEncoder(self.device) if self.model.use_audio else None
        self.face_encoder = FaceEncoder(self.device) if self.model.use_vision else None

    @torch.no_grad()
    def classify(self, text: str, prev_text: str, wave=None, faces=None) -> dict:
        inputs = {}
        if self.tokenizer:
            inputs["tokens"] = {k: v.to(self.device) for k, v in tokenize(self.tokenizer, [text], [prev_text] if self.context else None).items()}
        if self.audio_encoder:
            present = wave is not None
            inputs["audio"] = (self.audio_encoder.embed(wave)[None] if present else torch.zeros(1, AUDIO_LAYERS, AUDIO_DIM)).to(self.device)
            inputs["audio_present"] = torch.tensor([present], device=self.device)
        if self.face_encoder:
            present = bool(faces)
            inputs["vision"] = (self.face_encoder.embed(faces)[None] if present else torch.zeros(1, VISION_DIM)).to(self.device)
            inputs["vision_present"] = torch.tensor([present], device=self.device)
        logits = self.model(**inputs)
        probs = self.model.probabilities(logits)[0].cpu()
        views = {m: EMOTIONS[logits[m].argmax()] if inputs.get(f"{m}_present", torch.tensor([True])).item() else None for m in self.model.modalities}
        opinions = [v for v in views.values() if v]
        views["agree"] = len(set(opinions)) == 1 if len(opinions) >= 2 else None
        confidence = round(probs.max().item(), 3)
        return {
            "emotion": EMOTIONS[probs.argmax()],
            "confidence": confidence,
            "certainty": certainty(confidence),
            "probs": {e: round(p, 3) for e, p in zip(EMOTIONS, probs.tolist())},
            "views": views,
        }


def certainty(confidence: float) -> str:
    if confidence >= CERTAINTY_THRESHOLDS["high"]:
        return "high"
    return "medium" if confidence >= CERTAINTY_THRESHOLDS["medium"] else "low"


class Session:
    """Dialogue memory plus the event stream of one turn: state -> token* -> done."""

    def __init__(self, classifier: Classifier, responder, session_id: str = "s1"):
        self.classifier = classifier
        self.responder = responder
        self.session_id = session_id
        self.memory: deque[dict] = deque(maxlen=MEMORY_MESSAGES)  # {"speaker": "user"|"robot", "text", "emotion" (user only)}
        self.turn = 0

    def step(self, text: str, audio_path: str | None = None, video: str | None = None) -> Iterator[dict]:
        """One endpointed utterance in (text, optional 16 kHz clip, optional video file or directory of face crops);
        yields the state event, then reply pieces, then the done event."""
        start = time.perf_counter()

        def elapsed_ms() -> int:
            return round((time.perf_counter() - start) * 1000)

        # Inputs are loaded before the turn counts, so unreadable input raises without side effects.
        wave = load_audio(audio_path) if audio_path and self.classifier.audio_encoder else None
        faces = None
        if video and not Path(video).exists():
            raise FileNotFoundError(video)
        if video and self.classifier.face_encoder:
            faces = load_faces(video) if Path(video).is_dir() else faces_from_video(video)
        self.turn += 1
        prev_text = self.memory[-1]["text"] if self.memory else ""
        state = {
            "event": "state",
            "session_id": self.session_id,
            "turn": self.turn,
            "text": text,
            "audio_seconds": round(len(wave) / SAMPLE_RATE, 2) if wave is not None else None,
            "faces": len(faces) if faces is not None else None,
            **self.classifier.classify(text, prev_text, wave, faces),
            "context": list(self.memory),
            "latency_ms": {"state": elapsed_ms()},
        }
        yield state
        self.memory.append({"speaker": "user", "text": text, "emotion": state["emotion"]})

        messages = render_messages(state)
        pieces, first_token_ms, error = [], None, None
        try:
            for piece in self.responder.stream(messages):
                if not piece:  # the streamer flushes empty strings while it buffers partial words
                    continue
                first_token_ms = first_token_ms or elapsed_ms()
                pieces.append(piece)
                yield {"event": "token", "session_id": self.session_id, "turn": self.turn, "text": piece}
            reply, source = finish(state["emotion"], "".join(pieces))
        except Exception as failure:  # the robot still gets a line; the failure is reported in the event
            reply, source, error = FALLBACK[state["emotion"]], "fallback", f"{type(failure).__name__}: {failure}"
        self.memory.append({"speaker": "robot", "text": reply})
        yield {
            "event": "done",
            "session_id": self.session_id,
            "turn": self.turn,
            "response": reply,
            "source": source,
            "prompt": messages[-1]["content"],
            "latency_ms": {"state": state["latency_ms"]["state"], "first_token": first_token_ms, "response": elapsed_ms()},
            **({"error": error} if error else {}),
        }


def parameter_counts(classifier: Classifier, responder) -> dict[str, float]:
    """Millions of parameters on the inference path (the challenge caps the total at 6 000)."""
    counts = {"classifier": count_parameters(classifier.model), "responder": responder.parameter_count()}
    if classifier.face_encoder:
        counts["face_encoder"] = count_parameters(classifier.face_encoder.model)
        counts["face_detector"] = FACE_DETECTOR_PARAMS
    if classifier.audio_encoder:
        counts["audio_encoder"] = count_parameters(classifier.audio_encoder.model)
    counts["total"] = sum(counts.values())
    return {k: round(v / 1e6, 2) for k, v in counts.items()}
