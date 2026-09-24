"""EmotionModel: fine-tuned text encoder + frozen-feature audio branch + a small fusion head."""

import torch
from torch import nn
from transformers import AutoModel

from emo.config import AUDIO_DIM, AUDIO_LAYERS, EMOTIONS, HIDDEN_DIM, TEXT_MODEL

N_CLASSES = len(EMOTIONS)
TEXT_DIM = 768


class EmotionModel(nn.Module):
    """Predicts a MELD emotion from text tokens and/or cached WavLM features.

    modalities: "text", "audio" or "both". Every branch has its own head, so a "both" model also
    reports what the words alone and the voice alone suggest; "both" adds the fused head on top.
    forward() returns one logits tensor per head; `primary` names the head used for the prediction.
    """

    def __init__(self, modalities: str = "both"):
        super().__init__()
        self.modalities = modalities
        self.use_text = modalities in ("text", "both")
        self.use_audio = modalities in ("audio", "both")
        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
            self.text_dropout = nn.Dropout(0.1)
            self.text_head = nn.Linear(TEXT_DIM, N_CLASSES)
        if self.use_audio:
            self.layer_weights = nn.Parameter(torch.zeros(AUDIO_LAYERS))  # which WavLM layers carry emotion
            self.audio_proj = nn.Sequential(nn.LayerNorm(AUDIO_DIM), nn.Linear(AUDIO_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1))
            self.audio_head = nn.Linear(HIDDEN_DIM, N_CLASSES)
        if modalities == "both":
            self.fused_head = nn.Sequential(
                nn.Linear(TEXT_DIM + HIDDEN_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, N_CLASSES)
            )
        self.register_buffer("temperature", torch.ones(1))  # fitted on dev after training

    @property
    def primary(self) -> str:
        return "fused" if self.modalities == "both" else self.modalities

    def forward(self, tokens: dict | None = None, audio: torch.Tensor | None = None, audio_present: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """tokens: tokenizer output; audio: [B, 13, 768]; audio_present: [B] bool (False -> audio is ignored)."""
        logits = {}
        if self.use_text:
            text_vec = self.text_dropout(self.text_encoder(**tokens).last_hidden_state[:, 0])  # the <s> token
            logits["text"] = self.text_head(text_vec)
        if self.use_audio:
            layer_mix = torch.softmax(self.layer_weights, dim=0)
            audio_vec = self.audio_proj((layer_mix[:, None] * audio).sum(dim=1))
            audio_vec = audio_vec.masked_fill(~audio_present[:, None], 0.0)
            logits["audio"] = self.audio_head(audio_vec)
        if self.modalities == "both":
            logits["fused"] = self.fused_head(torch.cat([text_vec, audio_vec], dim=1))
        return logits

    def probabilities(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calibrated class probabilities of the primary head."""
        return torch.softmax(logits[self.primary] / self.temperature, dim=-1)


def save(model: EmotionModel, path) -> None:
    """Half-precision checkpoint (~165 MB for a text-bearing model)."""
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)


def load(path, modalities: str, device: torch.device) -> EmotionModel:
    model = EmotionModel(modalities)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
