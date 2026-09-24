"""One conversation with the robot: classify each utterance, emit a state event, then stream a grounded reply."""

import json
import time
from collections import deque
from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from emo.audio import AudioEncoder, load_audio
from emo.config import AUDIO_DIM, AUDIO_LAYERS, CERTAINTY_THRESHOLDS, DEPLOYED_RUN, EMOTIONS, MEMORY_MESSAGES, RUNS_DIR, SAMPLE_RATE, TEXT_MODEL, pick_device
from emo.data import tokenize
from emo.model import count_parameters, load
from emo.responder import FALLBACK, Responder, finish, render_messages


class Classifier:
    """A trained run, ready for one utterance at a time: (text, previous text, audio) -> emotion fields."""

    def __init__(self, run: str = DEPLOYED_RUN, device: torch.device | None = None):
        self.device = device or pick_device()
        info = json.loads((RUNS_DIR / run / "run.json").read_text())
        self.context = bool(info["context"])
        self.model = load(RUNS_DIR / run / "model.pt", info["modalities"], self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL) if self.model.use_text else None
        self.audio_encoder = AudioEncoder(self.device) if self.model.use_audio else None

    @torch.no_grad()
    def classify(self, text: str, prev_text: str, wave) -> dict:
        tokens = None
        if self.tokenizer:
            tokens = {k: v.to(self.device) for k, v in tokenize(self.tokenizer, [text], [prev_text] if self.context else None).items()}
        present = self.audio_encoder is not None and wave is not None
        audio = self.audio_encoder.embed(wave)[None] if present else torch.zeros(1, AUDIO_LAYERS, AUDIO_DIM)
        logits = self.model(tokens, audio.to(self.device), torch.tensor([present], device=self.device))
        probs = self.model.probabilities(logits)[0].cpu()
        views = {
            "text": EMOTIONS[logits["text"].argmax()] if "text" in logits else None,
            "audio": EMOTIONS[logits["audio"].argmax()] if present and "audio" in logits else None,
        }
        views["agree"] = views["text"] == views["audio"] if views["text"] and views["audio"] else None
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

    def __init__(self, classifier: Classifier, responder: Responder, session_id: str = "s1"):
        self.classifier = classifier
        self.responder = responder
        self.session_id = session_id
        self.memory: deque[dict] = deque(maxlen=MEMORY_MESSAGES)  # {"speaker": "user"|"robot", "text", "emotion" (user only)}
        self.turn = 0

    def step(self, text: str, audio_path: str | None = None) -> Iterator[dict]:
        """One endpointed utterance in; yields the state event, then reply pieces, then the done event."""
        start = time.perf_counter()

        def elapsed_ms() -> int:
            return round((time.perf_counter() - start) * 1000)

        wave = load_audio(audio_path) if audio_path else None  # raises on unreadable input, before the turn counts
        self.turn += 1
        prev_text = self.memory[-1]["text"] if self.memory else ""
        state = {
            "event": "state",
            "session_id": self.session_id,
            "turn": self.turn,
            "text": text,
            "audio_seconds": round(len(wave) / SAMPLE_RATE, 2) if wave is not None else None,
            **self.classifier.classify(text, prev_text, wave),
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


def parameter_counts(classifier: Classifier, responder: Responder) -> dict[str, float]:
    """Millions of parameters on the inference path (the challenge caps the total at 6 000)."""
    counts = {"classifier": count_parameters(classifier.model), "responder": count_parameters(responder.model)}
    if classifier.audio_encoder:
        counts["audio_encoder"] = count_parameters(classifier.audio_encoder.model)
    counts["total"] = sum(counts.values())
    return {k: round(v / 1e6, 2) for k, v in counts.items()}
