"""Audio: load 16 kHz clips and turn them into frozen WavLM features (live or cached)."""

import argparse
import os
import shutil
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

from emo.config import AUDIO_MODEL, DATA_DIR, MAX_AUDIO_SECONDS, MIN_AUDIO_SECONDS, SAMPLE_RATE, pick_device
from emo.data import load_manifest


def load_audio(path: str) -> np.ndarray | None:
    """Mono float32 samples at 16 kHz, cropped to MAX_AUDIO_SECONDS. None when the clip is too short to use."""
    wave, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz audio, got {rate} Hz. Convert with: ffmpeg -i {path} -ac 1 -ar {SAMPLE_RATE} out.wav")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if len(wave) < MIN_AUDIO_SECONDS * SAMPLE_RATE:
        return None
    return wave[: int(MAX_AUDIO_SECONDS * SAMPLE_RATE)]


class AudioEncoder:
    """Frozen WavLM-base-plus. embed(wave) -> [13, 768]: every hidden state averaged over time."""

    def __init__(self, device: torch.device):
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        self.model = AutoModel.from_pretrained(AUDIO_MODEL).to(device).eval()

    @torch.no_grad()
    def embed(self, wave: np.ndarray) -> torch.Tensor:
        inputs = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states  # 13 x [1, frames, 768]
        return torch.stack([h[0].mean(dim=0) for h in hidden_states]).cpu()


def features_path(split: str):
    return DATA_DIR / f"features_audio_{split}.pt"


def load_features(split: str) -> dict[str, torch.Tensor]:
    """id -> [13, 768] fp16 features for every manifest row that has usable audio."""
    return torch.load(features_path(split))


def build_feature_cache(device: torch.device) -> None:
    """Embed every clip once, one clip at a time (no padding), so cached and live features are identical.

    Smallest split first, so a problem shows up early; each file is written atomically.
    """
    if shutil.disk_usage(DATA_DIR).free < 1e9:
        raise SystemExit("less than 1 GB of free disk: the feature cache needs ~300 MB, plus headroom for swap")
    encoder = AudioEncoder(device)
    for split in ("dev", "test", "train"):
        if features_path(split).exists():
            print(f"{split}: {features_path(split).name} exists, skipping")
            continue
        rows = [r for r in load_manifest(split) if r["has_audio"]]
        features, start = {}, time.perf_counter()
        for i, row in enumerate(rows, 1):
            features[row["id"]] = encoder.embed(load_audio(row["audio_path"])).half()
            if i % 1000 == 0 or i == len(rows):
                print(f"{split}: {i}/{len(rows)} clips in {time.perf_counter() - start:.0f} s", flush=True)
        partial = features_path(split).with_suffix(".tmp")
        torch.save(features, partial)
        os.replace(partial, features_path(split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache WavLM features for every MELD clip.")
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    build_feature_cache(pick_device(parser.parse_args().device))
