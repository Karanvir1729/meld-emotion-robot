"""EmotionModel contract: heads per modality set, calibrated probabilities, hidden inputs contribute nothing, view refits."""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from emo.config import AUDIO_DIM, AUDIO_LAYERS, EMOTIONS, TEXT_MODEL, VISION_DIM
from emo.data import Collate, MeldDataset, tokenize
from emo.model import EmotionModel, canonical
from emo.train import fit_view_heads


@pytest.fixture(scope="module")
def batch():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    return {
        "tokens": tokenize(tokenizer, ["I said I was fine.", "Great news!"], ["How are you?", ""]),
        "audio": torch.randn(2, AUDIO_LAYERS, AUDIO_DIM),
        "audio_present": torch.tensor([True, True]),
        "vision": torch.randn(2, VISION_DIM),
        "vision_present": torch.tensor([True, True]),
    }


def test_canonical_order_and_validation():
    assert canonical(["vision", "text"]) == ("text", "vision")
    with pytest.raises(ValueError):
        canonical(["text", "smell"])


@pytest.mark.parametrize("modalities, heads", [
    (("text",), {"text"}), (("audio",), {"audio"}), (("vision",), {"vision"}),
    (("text", "vision"), {"text", "vision", "fused"}), (("text", "audio", "vision"), {"text", "audio", "vision", "fused"}),
])
def test_heads_and_probabilities(batch, modalities, heads):
    model = EmotionModel(modalities).eval()
    with torch.no_grad():
        logits = model(**batch)
    assert set(logits) == heads and all(l.shape == (2, len(EMOTIONS)) for l in logits.values())
    model.temperature.fill_(2.0)
    probs = model.probabilities(logits)
    assert torch.allclose(probs.sum(1), torch.ones(2))
    assert torch.allclose(probs, torch.softmax(logits[model.primary] / 2.0, dim=-1))


@pytest.mark.parametrize("modality", ["audio", "vision"])
def test_hidden_input_is_ignored(batch, modality):
    model = EmotionModel((modality,)).eval()
    absent = torch.tensor([False, False])
    with torch.no_grad():
        a = model(**{modality: batch[modality], f"{modality}_present": absent})[modality]
        b = model(**{modality: torch.zeros_like(batch[modality]), f"{modality}_present": absent})[modality]
    assert torch.allclose(a, b)  # whatever the features, an absent input gives the same (prior) logits


def test_refitting_view_heads_keeps_the_fused_prediction():
    """fit_view_heads only trains the per-modality heads; the fused head's output must not move."""
    torch.manual_seed(0)
    rows = [{"id": f"u{i}", "emotion": EMOTIONS[i % 7], "has_audio": True, "has_vision": i % 3 != 0} for i in range(64)]
    features = {"audio": {r["id"]: torch.randn(AUDIO_LAYERS, AUDIO_DIM) for r in rows}, "vision": {r["id"]: torch.randn(VISION_DIM) for r in rows}}
    loader = DataLoader(MeldDataset(rows, features), batch_size=16, collate_fn=Collate(None, context=False))
    model = EmotionModel(("audio", "vision")).eval()
    batch = next(iter(loader))
    inputs = {k: batch[k] for k in ("audio", "audio_present", "vision", "vision_present")}
    with torch.no_grad():
        before = model(**inputs)
    scores = fit_view_heads(model, {"train": loader, "dev": loader}, torch.device("cpu"))
    with torch.no_grad():
        after = model.eval()(**inputs)
    assert set(scores) == {"audio", "vision"}
    assert torch.allclose(before["fused"], after["fused"])
    assert not torch.allclose(before["vision"], after["vision"])
