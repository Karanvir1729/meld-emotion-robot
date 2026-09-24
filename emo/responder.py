"""Reply generation: a small instruct LLM prompted only from the emotion state, with a hand-written fallback."""

import re
import threading
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from emo.config import MAX_NEW_TOKENS, MAX_REPLY_WORDS, RESPONSE_MODEL

SYSTEM = (
    "You are Pip, a small, warm, curious robot companion talking with one person. Reply in one or two short "
    "sentences of plain spoken English, at most 30 words. No emojis, lists or stage directions. Every message "
    "from the person starts with a note in [brackets] about how they sound: let it shape your tone, and never "
    "mention the note, feelings analysis, or being an AI."
)

# The whole character policy: how the person sounds, and how Pip behaves, for each MELD emotion.
TONE = {"neutral": "calm", "joy": "happy", "sadness": "sad", "anger": "angry", "fear": "scared", "disgust": "disgusted", "surprise": "surprised"}
STYLE = {
    "neutral": "keep it light and ask a small follow-up question",
    "joy": "share the good mood, be warm and a little playful",
    "sadness": "be gentle and brief, acknowledge the feeling, do not try to fix it",
    "anger": "stay calm and steady, acknowledge the frustration, no jokes, no arguing",
    "fear": "be reassuring and concrete",
    "disgust": "acknowledge it matter-of-factly, do not pile on",
    "surprise": "match the surprise and be curious about what happened",
}

# Used when the model's reply fails validation, so the robot always has something to say.
FALLBACK = {
    "neutral": "Got it. What else is going on today?",
    "joy": "That's lovely to hear! Tell me more.",
    "sadness": "I'm sorry. I'm here if you want to talk about it.",
    "anger": "That sounds really frustrating. Do you want to tell me what happened?",
    "fear": "That sounds scary. I'm right here with you.",
    "disgust": "Yikes, that does sound unpleasant.",
    "surprise": "Wow, really? What happened?",
}

META_LANGUAGE = re.compile(r"analy[sz]|detect|confidence|probabilit|label|\bAI\b|language model|assistant|\[", re.IGNORECASE)


def render_messages(state: dict) -> list[dict]:
    """The chat sent to the LLM: persona, the remembered turns, then the utterance with its note. A function of the state only."""
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in state["context"]:
        if turn["speaker"] == "robot":
            messages.append({"role": "assistant", "content": turn["text"]})
        else:
            messages.append({"role": "user", "content": f"[sounded {TONE[turn['emotion']]}] {turn['text']}"})
    messages.append({"role": "user", "content": f"[{note(state)}] {state['text']}"})
    return messages


def note(state: dict) -> str:
    """The stage note for one utterance: how the person sounds and how Pip should respond."""
    emotion, views = state["emotion"], state["views"]
    if state["certainty"] == "low":
        parts = ["hard to tell how they feel, ask rather than assume"]
    else:
        parts = [f"sounds {TONE[emotion]}"]
        if views["agree"] is False:
            parts.append(f"the words alone read {TONE[views['text']]} but the voice sounds {TONE[views['audio']]}, you may gently notice the mismatch")
    parts.append(STYLE[emotion])
    return "; ".join(parts)


def finish(emotion: str, raw: str) -> tuple[str, str]:
    """Cleans a streamed reply and validates it. Returns (reply, source) with source "llm" or "fallback"."""
    reply = raw.strip().strip('"').removeprefix("Pip:").strip()
    if 1 <= len(reply.split()) <= MAX_REPLY_WORDS and not META_LANGUAGE.search(reply):
        return reply, "llm"
    return FALLBACK[emotion], "fallback"


class Responder:
    """Qwen2.5-0.5B-Instruct, frozen. stream(messages) yields the reply as it is generated, one line only."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(RESPONSE_MODEL)
        dtype = torch.bfloat16 if device.type == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(RESPONSE_MODEL, dtype=dtype).to(device).eval()

    def stream(self, messages: list[dict], sample: bool = True) -> Iterator[str]:
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        settings = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, **settings, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.1, stop_strings=["\n"], tokenizer=self.tokenizer),
        )
        thread.start()
        for piece in streamer:
            yield piece.split("\n")[0]  # everything after a newline was a second paragraph
        thread.join()
