"""Vision: a frozen facial-expression encoder over the face crops (live or cached). Crops come from emo.faces."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import f1_score
from transformers import ViTForImageClassification

from emo.config import DATA_DIR, EMOTIONS, RUNS_DIR, VISION_MODEL, pick_device
from emo.data import load_manifest
from emo.faces import load_faces


class FaceEncoder:
    """Frozen ViT fine-tuned on facial expressions. embed(crops) -> [768]: the CLS vector averaged over the crops."""

    def __init__(self, device: torch.device):
        self.device = device
        config = json.loads(Path(hf_hub_download(VISION_MODEL, "preprocessor_config.json")).read_text())
        self.size = config["size"] if isinstance(config["size"], int) else config["size"]["height"]
        self.mean, self.std = np.array(config["image_mean"], dtype=np.float32), np.array(config["image_std"], dtype=np.float32)
        self.model = ViTForImageClassification.from_pretrained(VISION_MODEL).to(device).eval()

    @torch.no_grad()
    def embed(self, crops: list[Image.Image]) -> torch.Tensor:
        return self.model.vit(pixel_values=self.pixels(crops)).last_hidden_state[:, 0].mean(dim=0).cpu()

    def pixels(self, crops: list[Image.Image]) -> torch.Tensor:
        """The model's own preprocessing (resize, scale to [0, 1], normalise), done here: AutoImageProcessor would need torchvision."""
        pixels = np.stack([np.asarray(c.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255 for c in crops])
        return torch.from_numpy((pixels - self.mean) / self.std).permute(0, 3, 1, 2).to(self.device)


# --- feature cache ------------------------------------------------------------

def features_path(split: str):
    return DATA_DIR / f"features_vision_{split}.pt"


def load_features(split: str) -> dict[str, torch.Tensor]:
    """id -> [768] fp16 features for every manifest row with at least one face crop."""
    return torch.load(features_path(split))


def build_feature_cache(device: torch.device) -> None:
    """Embed every utterance's face crops once; smallest split first, each file written atomically."""
    if shutil.disk_usage(DATA_DIR).free < 1e9:
        raise SystemExit("less than 1 GB of free disk")
    encoder = FaceEncoder(device)
    for split in ("dev", "test", "train"):
        if features_path(split).exists():
            print(f"{split}: {features_path(split).name} exists, skipping")
            continue
        rows = [r for r in load_manifest(split) if r["has_vision"]]
        features, start = {}, time.perf_counter()
        for i, row in enumerate(rows, 1):
            features[row["id"]] = encoder.embed(load_faces(row["faces_dir"])).half()
            if i % 1000 == 0 or i == len(rows):
                print(f"{split}: {i}/{len(rows)} clips in {time.perf_counter() - start:.0f} s", flush=True)
        partial = features_path(split).with_suffix(".tmp")
        torch.save(features, partial)
        os.replace(partial, features_path(split))


def zero_shot_check(device: torch.device, split: str = "dev") -> dict:
    """How much emotion the face alone carries: the expression model's own classifier, mapped to MELD labels, with no training."""
    fer_to_meld = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "joy", "neutral": "neutral", "sad": "sadness", "surprise": "surprise"}
    encoder = FaceEncoder(device)
    labels, preds = [], []
    for row in load_manifest(split):
        if row["has_vision"]:
            logits = encoder.model(pixel_values=encoder.pixels(load_faces(row["faces_dir"]))).logits
            preds.append(EMOTIONS.index(fer_to_meld[encoder.model.config.id2label[int(logits.softmax(-1).mean(0).argmax())]]))
            labels.append(EMOTIONS.index(row["emotion"]))
    labels, preds = np.array(labels), np.array(preds)
    result = {
        "split": split,
        "clips_with_a_face": len(labels),
        "zero_shot_weighted_f1": round(f1_score(labels, preds, average="weighted"), 3),
        "majority_weighted_f1": round(f1_score(labels, np.zeros_like(labels), average="weighted"), 3),
        "joy_read_as_happy": round(float(np.mean(preds[labels == 1] == 1)), 3),
        "neutral_read_as_happy": round(float(np.mean(preds[labels == 0] == 1)), 3),
    }
    (RUNS_DIR / f"vision_zero_shot_{split}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache facial-expression features for every MELD clip with a face.")
    parser.add_argument("--device", help="mps or cpu (default: mps when available)")
    parser.add_argument("--zero-shot", action="store_true", help="instead, score the expression model's own classifier on dev faces")
    args = parser.parse_args()
    if args.zero_shot:
        with torch.no_grad():
            print(json.dumps(zero_shot_check(pick_device(args.device)), indent=2))
    else:
        build_feature_cache(pick_device(args.device))
