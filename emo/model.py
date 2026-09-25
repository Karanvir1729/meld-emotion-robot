"""EmotionModel: fine-tuned text encoder + frozen-feature audio / vision branches + a small fusion head."""

import torch
from torch import nn
from transformers import AutoModel

from emo.config import AUDIO_DIM, AUDIO_LAYERS, EMOTIONS, HIDDEN_DIM, MODALITIES, TEXT_MODEL, VISION_DIM

N_CLASSES = len(EMOTIONS)
TEXT_DIM = 768


def canonical(modalities) -> tuple[str, ...]:
    """Validated modalities in the canonical order, e.g. ("text", "vision")."""
    chosen = set(modalities)
    unknown = chosen - set(MODALITIES)
    if unknown or not chosen:
        raise ValueError(f"modalities must be a non-empty subset of {MODALITIES}, got {sorted(chosen)}")
    return tuple(m for m in MODALITIES if m in chosen)


def primary_head(modalities: tuple[str, ...]) -> str:
    """The head whose prediction is the model's answer."""
    return "fused" if len(modalities) > 1 else modalities[0]


class EmotionModel(nn.Module):
    """Predicts a MELD emotion from any subset of {text tokens, cached WavLM features, cached face features}.

    Every branch has its own head, so a fused model also reports what the words, the voice and the face each
    suggest (the `views`); with more than one branch a fused head on the concatenated branch vectors gives the answer.
    forward() returns one logits tensor per head; `primary` names the head used for the prediction.
    """

    def __init__(self, modalities=("text", "vision")):
        super().__init__()
        self.modalities = canonical(modalities)
        self.use_text = "text" in self.modalities
        self.use_audio = "audio" in self.modalities
        self.use_vision = "vision" in self.modalities
        fused_in = 0
        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
            self.text_dropout = nn.Dropout(0.1)
            self.text_head = nn.Linear(TEXT_DIM, N_CLASSES)
            fused_in += TEXT_DIM
        if self.use_audio:
            self.layer_weights = nn.Parameter(torch.zeros(AUDIO_LAYERS))  # which WavLM layers carry emotion
            self.audio_proj = nn.Sequential(nn.LayerNorm(AUDIO_DIM), nn.Linear(AUDIO_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1))
            self.audio_head = nn.Linear(HIDDEN_DIM, N_CLASSES)
            fused_in += HIDDEN_DIM
        if self.use_vision:
            self.vision_proj = nn.Sequential(nn.LayerNorm(VISION_DIM), nn.Linear(VISION_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1))
            self.vision_head = nn.Linear(HIDDEN_DIM, N_CLASSES)
            fused_in += HIDDEN_DIM
        if len(self.modalities) > 1:
            self.fused_head = nn.Sequential(nn.Linear(fused_in, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, N_CLASSES))
        self.register_buffer("temperature", torch.ones(1))  # fitted on dev after training

    @property
    def primary(self) -> str:
        return primary_head(self.modalities)

    def forward(self, tokens=None, audio=None, audio_present=None, vision=None, vision_present=None) -> dict[str, torch.Tensor]:
        """tokens: tokenizer output; audio: [B, 13, 768]; vision: [B, 768]; *_present: [B] bool (False -> that input is ignored)."""
        vectors = self.branch_vectors(tokens, audio, audio_present, vision, vision_present)
        logits = {modality: getattr(self, f"{modality}_head")(vector) for modality, vector in vectors.items()}
        if len(vectors) > 1:
            logits["fused"] = self.fused_head(torch.cat(list(vectors.values()), dim=1))
        return logits

    def branch_vectors(self, tokens=None, audio=None, audio_present=None, vision=None, vision_present=None) -> dict[str, torch.Tensor]:
        """One vector per modality, in canonical order; an absent input gives a zero vector."""
        vectors = {}
        if self.use_text:
            vectors["text"] = self.text_dropout(self.text_encoder(**tokens).last_hidden_state[:, 0])  # the <s> token
        if self.use_audio:
            layer_mix = torch.softmax(self.layer_weights, dim=0)
            vectors["audio"] = self.audio_proj((layer_mix[:, None] * audio).sum(dim=1)).masked_fill(~audio_present[:, None], 0.0)
        if self.use_vision:
            vectors["vision"] = self.vision_proj(vision).masked_fill(~vision_present[:, None], 0.0)
        return vectors

    def probabilities(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calibrated class probabilities of the primary head."""
        return torch.softmax(logits[self.primary] / self.temperature, dim=-1)


def save(model: EmotionModel, path) -> None:
    """Half-precision checkpoint (~165 MB for a text-bearing model)."""
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)


def load(path, modalities, device: torch.device) -> EmotionModel:
    model = EmotionModel(modalities)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
