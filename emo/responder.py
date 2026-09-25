"""Reply generation: a small instruct LLM prompted only from the emotion state, with a hand-written fallback."""

import re
import threading
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from emo.config import MAX_NEW_TOKENS, MAX_REPLY_WORDS, RESPONSE_MODEL

SYSTEM = (
    "You are Pip, a small, warm, curious robot friend chatting with one person. You are a friend, not an "
    "assistant: never offer help, services or advice, never ask how you can help, never apologise for not "
    "understanding. Talk the way a close friend talks: react to what they said, in one or two short sentences "
    f"of plain spoken English, at most {MAX_REPLY_WORDS} words. No emojis, lists or stage directions. Every "
    "message from the person starts with a note in [brackets] about how they seem: let it shape your tone, "
    "and never mention the note, feelings analysis, or being an AI."
)

# The whole character policy: how the person seems, and how Pip behaves, for each MELD emotion.
TONE = {"neutral": "calm", "joy": "happy", "sadness": "sad", "anger": "angry", "fear": "scared", "disgust": "disgusted", "surprise": "surprised"}
VIEW = {"text": "the words alone read {}", "audio": "the voice sounds {}", "vision": "the face looks {}"}
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

# Out of character: talk about the analysis, or assistant-speak ("I can't assist with that request").
OUT_OF_CHARACTER = re.compile(
    r"analy[sz]|detect|confidence|probabilit|label|\bAI\b|language model|assist|capabilit|my purpose|how can I help|anything (else|specific) (I can|you'?d like)|\[",
    re.IGNORECASE,
)

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


def render_messages(state: dict) -> list[dict]:
    """The chat sent to the LLM: persona, the remembered turns, then the utterance with its note. A function of the state only."""
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in state["context"]:
        if turn["speaker"] == "robot":
            messages.append({"role": "assistant", "content": turn["text"]})
        else:
            messages.append({"role": "user", "content": f"[seemed {TONE[turn['emotion']]}] {turn['text']}"})
    messages.append({"role": "user", "content": f"[{note(state)}] {state['text']}"})
    return messages


def note(state: dict) -> str:
    """The stage note for one utterance: how the person seems, which view disagrees, and how Pip should respond.

    A view is only mentioned when it shows a different, non-neutral emotion: a face that shows nothing is the
    absence of a signal, not evidence against an angry sentence.
    """
    emotion, views = state["emotion"], state["views"]
    if state["certainty"] == "low":
        return "hard to tell how they feel, ask rather than assume"
    parts = [f"seems {TONE[emotion]}"]
    mismatches = [VIEW[m].format(TONE[v]) for m, v in views.items() if m != "agree" and v not in (None, emotion, "neutral")]
    if mismatches:
        parts.append(", ".join(mismatches) + ", you may gently notice the mismatch")
    parts.append(STYLE[emotion])
    return "; ".join(parts)


def finish(emotion: str, raw: str) -> tuple[str, str]:
    """Cleans a streamed reply and validates it. Returns (reply, source) with source "llm" or "fallback"."""
    reply = " ".join(EMOJI.sub("", raw).split()).strip('"').removeprefix("Pip:").strip()  # emojis cannot be spoken
    if 1 <= len(reply.split()) <= MAX_REPLY_WORDS and not OUT_OF_CHARACTER.search(reply):
        return reply, "llm"
    return FALLBACK[emotion], "fallback"


def load_responder(device: torch.device):
    """The reply model named by RESPONSE_MODEL: 4-bit MLX weights (mlx-community/...) on Apple silicon, else transformers."""
    return MlxResponder() if RESPONSE_MODEL.startswith("mlx-community/") else Responder(device)


class Responder:
    """A HuggingFace chat model, frozen. stream(messages) yields the reply as it is generated, one line only."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(RESPONSE_MODEL)
        dtype = torch.bfloat16  # 6 GB for the 3B model instead of 12 GB in fp32
        self.model = AutoModelForCausalLM.from_pretrained(RESPONSE_MODEL, dtype=dtype).to(device).eval()

    def stream(self, messages: list[dict], sample: bool = True) -> Iterator[str]:
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        settings = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        failure = []

        def generate() -> None:
            try:
                self.model.generate(**inputs, **settings, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.1, stop_strings=["\n"], tokenizer=self.tokenizer)
            except Exception as error:  # otherwise the streamer never ends and the caller waits forever
                failure.append(error)
                streamer.end()

        thread = threading.Thread(target=generate)
        thread.start()
        for piece in streamer:
            yield piece.split("\n")[0]  # everything after a newline was a second paragraph
        thread.join()
        if failure:
            raise failure[0]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class MlxResponder:
    """The same contract on Apple's MLX runtime, for 4-bit quantized weights (always on the Apple GPU).

    The persona is the same in every prompt, so its key/value cache is computed once and every turn only
    prefills the conversation after it (~130 fewer tokens, ~0.3 s sooner to the first word).
    """

    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.model, self.tokenizer = load(RESPONSE_MODEL)
        self.persona = self.tokenizer.encode(self.tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}], tokenize=False))
        self.cache = make_prompt_cache(self.model)
        self.model(mx.array(self.persona)[None], cache=self.cache)
        mx.eval([layer.state for layer in self.cache])

    def stream(self, messages: list[dict], sample: bool = True) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tokens = self.tokenizer.encode(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
        assert tokens[: len(self.persona)] == self.persona, "every prompt starts with the persona"
        trim_prompt_cache(self.cache, self.cache[0].offset - len(self.persona))  # forget the previous turn, keep the persona
        sampler = make_sampler(temp=0.7, top_p=0.9) if sample else make_sampler(temp=0.0)
        processors = make_logits_processors(repetition_penalty=1.1)
        for step in stream_generate(self.model, self.tokenizer, tokens[len(self.persona):], prompt_cache=self.cache,
                                    max_tokens=MAX_NEW_TOKENS, sampler=sampler, logits_processors=processors):
            if "\n" in step.text:  # breaking out stops generation: the generator is lazy
                yield step.text.split("\n")[0]
                break
            yield step.text

    def parameter_count(self) -> int:
        """Logical parameters: a quantized layer packs 32 // bits weights into each uint32 (scales are not counted)."""
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        total = 0
        for _, layer in tree_flatten(self.model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)):
            if hasattr(layer, "bits"):  # QuantizedLinear / QuantizedEmbedding
                rows, packed = layer.weight.shape
                total += rows * packed * 32 // layer.bits + (layer["bias"].size if "bias" in layer else 0)
            else:
                total += sum(p.size for _, p in tree_flatten(layer.parameters()))
        return total
