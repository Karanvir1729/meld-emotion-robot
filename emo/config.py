"""Single source of truth: paths, labels, model ids and hyperparameters."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MELD_DIR = DATA_DIR / "meld"
RUNS_DIR = ROOT / "runs"

# --- MELD --------------------------------------------------------------------
SPLITS = ["train", "dev", "test"]
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
MELD_CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
MELD_AUDIO_URL = "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/{split}.tar.gz"  # 16 kHz mono FLAC per utterance

# --- Models (all run locally; 0.67 B parameters in total, cap is 6 B) --------
TEXT_MODEL = "distilbert/distilroberta-base"  # 82 M, fine-tuned
AUDIO_MODEL = "microsoft/wavlm-base-plus"  # 95 M, frozen feature extractor
RESPONSE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 494 M, frozen reply generator

# --- Audio -------------------------------------------------------------------
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0  # MELD median is 2.5 s; longer clips are cropped
MIN_AUDIO_SECONDS = 0.3  # shorter clips are treated as "no audio"
AUDIO_LAYERS = 13  # WavLM-base-plus hidden states: embeddings + 12 layers
AUDIO_DIM = 768

# --- Classifier training -----------------------------------------------------
MAX_TEXT_TOKENS = 96
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = {"text": 4, "both": 4, "audio": 30}  # the audio-only head is tiny and needs more passes
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
AUX_LOSS_WEIGHT = 0.3  # weight of the text-only and audio-only heads when training "both"
MODALITY_DROPOUT = 0.15  # fraction of training samples whose audio is hidden from the fused head

# --- Inference ---------------------------------------------------------------
DEPLOYED_RUN = "both"  # run directory used by demo / serve / bench (fixed before evaluation)
CERTAINTY_THRESHOLDS = {"high": 0.70, "medium": 0.45}  # on calibrated max-probability
MEMORY_TURNS = 4  # dialogue turns remembered per session
MAX_REPLY_WORDS = 30
MAX_NEW_TOKENS = 48
REALTIME_TARGETS_MS = {"state": 300, "first_token": 800, "response": 2500}  # p95 after the utterance ends; see README


def pick_device(requested: str | None = None) -> torch.device:
    """MPS (Apple GPU) when available, else CPU. `requested` overrides."""
    if requested:
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
