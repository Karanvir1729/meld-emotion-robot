"""EmotionModel contract: heads per modality, calibrated probabilities, hidden audio contributes nothing."""

import pytest
import torch
from transformers import AutoTokenizer

from emo.config import AUDIO_DIM, AUDIO_LAYERS, EMOTIONS, TEXT_MODEL
from emo.data import tokenize
from emo.model import EmotionModel


@pytest.fixture(scope="module")
def batch():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    return {
        "tokens": tokenize(tokenizer, ["I said I was fine.", "Great news!"], ["How are you?", ""]),
        "audio": torch.randn(2, AUDIO_LAYERS, AUDIO_DIM),
        "audio_present": torch.tensor([True, True]),
    }


@pytest.mark.parametrize("modalities, heads", [("text", {"text"}), ("audio", {"audio"}), ("both", {"text", "audio", "fused"})])
def test_heads_and_probabilities(batch, modalities, heads):
    model = EmotionModel(modalities).eval()
    with torch.no_grad():
        logits = model(batch["tokens"], batch["audio"], batch["audio_present"])
    assert set(logits) == heads and all(l.shape == (2, len(EMOTIONS)) for l in logits.values())
    model.temperature.fill_(2.0)
    probs = model.probabilities(logits)
    assert torch.allclose(probs.sum(1), torch.ones(2))
    assert torch.allclose(probs, torch.softmax(logits[model.primary] / 2.0, dim=-1))


def test_hidden_audio_is_ignored(batch):
    model = EmotionModel("audio").eval()
    absent = torch.tensor([False, False])
    with torch.no_grad():
        a = model(None, batch["audio"], absent)["audio"]
        b = model(None, torch.zeros_like(batch["audio"]), absent)["audio"]
    assert torch.allclose(a, b)  # whatever the features, absent audio gives the same (prior) logits
