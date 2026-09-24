"""Session event contract with stub models: event order, keys, dialogue memory, reply fallback."""

from emo.responder import FALLBACK
from emo.session import Session


class StubClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, text, prev_text, wave):
        self.calls.append((text, prev_text))
        return {"emotion": "anger", "confidence": 0.8, "certainty": "high", "probs": {}, "views": {"text": "neutral", "audio": "anger", "agree": False}}


class StubResponder:
    def __init__(self, pieces):
        self.pieces = pieces

    def stream(self, messages, sample=True):
        yield from self.pieces


def test_turn_emits_state_tokens_done_and_remembers_the_dialogue():
    classifier = StubClassifier()
    session = Session(classifier, StubResponder(["", "Hey,", "", " that sounds rough."]))  # empty pieces are dropped

    events = list(session.step("I said I was fine."))
    assert [e["event"] for e in events] == ["state", "token", "token", "done"]
    state, done = events[0], events[-1]
    assert state["emotion"] == "anger" and state["context"] == [] and state["audio_seconds"] is None
    assert done["response"] == "Hey, that sounds rough." and done["source"] == "llm"
    assert done["prompt"] == "[sounds angry; the words alone read calm but the voice sounds angry, you may gently notice the mismatch; stay calm and steady, acknowledge the frustration, no jokes, no arguing] I said I was fine."

    events = list(session.step("Whatever."))
    assert classifier.calls[1] == ("Whatever.", "Hey, that sounds rough.")  # previous turn is the classifier context
    assert events[0]["context"] == [
        {"speaker": "user", "text": "I said I was fine.", "emotion": "anger"},
        {"speaker": "robot", "text": "Hey, that sounds rough."},
    ]


def test_invalid_reply_falls_back_to_the_hand_written_line():
    session = Session(StubClassifier(), StubResponder(["As an AI language model I detect anger."]))
    done = list(session.step("Ugh."))[-1]
    assert (done["response"], done["source"]) == (FALLBACK["anger"], "fallback")


def test_generation_failure_still_ends_the_turn():
    class BrokenResponder:
        def stream(self, messages, sample=True):
            raise RuntimeError("out of memory")
            yield  # makes this a generator

    session = Session(StubClassifier(), BrokenResponder())
    events = list(session.step("Ugh."))
    assert [e["event"] for e in events] == ["state", "done"]
    assert events[-1]["source"] == "fallback" and events[-1]["error"] == "RuntimeError: out of memory"
    assert session.memory[-1] == {"speaker": "robot", "text": FALLBACK["anger"]}  # memory stays paired
