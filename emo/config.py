"""Single source of truth: paths, labels, model ids and hyperparameters."""

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MELD_DIR = DATA_DIR / "meld"
RUNS_DIR = ROOT / "runs"

# --- MELD --------------------------------------------------------------------
SPLITS = ["train", "dev", "test"]
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
MODALITIES = ["text", "audio", "vision"]  # canonical order everywhere (run names, fusion input)
MELD_CSV_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
MELD_AUDIO_URL = "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/{split}.tar.gz"  # 16 kHz mono FLAC per utterance
MELD_VIDEO_URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"  # 10.9 GB; streamed, only face crops are kept

# --- Models (all run locally; well under the 6 B parameter cap) --------------
TEXT_MODEL = "distilbert/distilroberta-base"  # 82 M, fine-tuned
VISION_MODEL = "trpakov/vit-face-expression"  # 86 M, frozen; ViT-base fine-tuned on facial expressions
AUDIO_MODEL = "microsoft/wavlm-base-plus"  # 94 M, frozen voice encoder
FACE_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_DETECTOR = DATA_DIR / "face_detection_yunet_2023mar.onnx"  # OpenCV YuNet
FACE_DETECTOR_PARAMS = 53121  # counted once from the ONNX initializers
FACES_DIR = MELD_DIR / "faces"  # data/meld/faces/{split}/{id}/{k}.jpg
VIDEOS_DIR = MELD_DIR / "videos"  # data/meld/videos/test/{id}.mp4, raw clips of a few dialogues
KEPT_VIDEO_DIALOGUES = range(30)  # test dialogues whose raw clips are kept, so demo and bench run the live video path (~0.2 GB)
# Qwen2.5-3B-Instruct (3.1 B), frozen: 4-bit through MLX on Apple silicon, bf16 through transformers elsewhere.
# EMO_RESPONSE_MODEL takes any HuggingFace chat model id (an mlx-community/... id selects the MLX backend).
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
RESPONSE_MODEL = os.environ.get("EMO_RESPONSE_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit" if APPLE_SILICON else "Qwen/Qwen2.5-3B-Instruct")

# --- Audio -------------------------------------------------------------------
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0  # MELD median is 2.5 s; longer clips are cropped
MIN_AUDIO_SECONDS = 0.3  # shorter clips are treated as "no audio"
MAX_UTTERANCE_SECONDS = 60  # longer "clips" are extraction errors (two in MELD test); also "no audio"
AUDIO_LAYERS = 13  # WavLM-base-plus hidden states: embeddings + 12 layers
AUDIO_DIM = 768

# --- Vision ------------------------------------------------------------------
FRAMES_PER_CLIP = 6  # face crops kept per utterance, evenly spaced over the first MAX_AUDIO_SECONDS
FACE_SIZE = 224  # crop side in pixels, what the ViT expects
VISION_DIM = 768

# --- Classifier training -----------------------------------------------------
MAX_TEXT_TOKENS = 96
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS_WITH_TEXT = 4  # fine-tuning the text encoder
EPOCHS_HEADS_ONLY = 30  # audio-only / vision-only heads are tiny and need more passes
LR_ENCODER = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
AUX_LOSS_WEIGHT = 0.3  # weight of each per-modality head in a fused run (they are refit afterwards, see train.fit_view_heads)
MODALITY_DROPOUT = 0.15  # fraction of training samples whose audio / vision is hidden from the fused head

# --- Inference ---------------------------------------------------------------
DEPLOYED_RUN = "text-audio-vision"  # run used by demo / serve / bench / responses: the three-modality extension; the core track is "text-vision"
CERTAINTY_THRESHOLDS = {"high": 0.70, "medium": 0.45}  # on calibrated max-probability
MEMORY_MESSAGES = 4  # remembered messages per session; the person and Pip alternate, so two exchanges
MAX_REPLY_WORDS = 30
MAX_NEW_TOKENS = 48
REALTIME_TARGETS_MS = {"state": 300, "first_token": 800, "response": 2500}  # p95 after the utterance ends; see README


def pick_device(requested: str | None = None):
    """CUDA, else MPS (Apple GPU), else CPU. `requested` overrides."""
    import torch  # local, so data-only processes (the face-extraction workers) never load torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
