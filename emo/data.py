"""MELD data: download, one manifest line per utterance, and the training Dataset/collate."""

import csv
import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from emo.config import (
    AUDIO_DIM,
    AUDIO_LAYERS,
    DATA_DIR,
    EMOTIONS,
    FACES_DIR,
    MAX_TEXT_TOKENS,
    MAX_UTTERANCE_SECONDS,
    MELD_AUDIO_URL,
    MELD_CSV_URL,
    MELD_DIR,
    MIN_AUDIO_SECONDS,
    SPLITS,
    VIDEOS_DIR,
    VISION_DIM,
)

FEATURE_SHAPES = {"audio": (AUDIO_LAYERS, AUDIO_DIM), "vision": (VISION_DIM,)}  # cached features per modality


def download() -> None:
    """Fetch the official CSVs (GitHub) and the 16 kHz FLAC clips (HuggingFace). Skips what already exists."""
    MELD_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        csv_path = MELD_DIR / f"{split}.csv"
        if not csv_path.exists():
            urllib.request.urlretrieve(MELD_CSV_URL.format(split=split), csv_path)
        audio_dir = MELD_DIR / "audio" / split
        if not audio_dir.exists():
            archive, _ = urllib.request.urlretrieve(MELD_AUDIO_URL.format(split=split))
            with tarfile.open(archive) as tar:
                tar.extractall(MELD_DIR / "audio", filter="data")  # creates audio/{split}/dia{D}_utt{U}.flac
        print(f"{split}: {csv_path.name} and {len(list(audio_dir.glob('*.flac')))} clips present")


def build_manifests() -> None:
    """CSV rows -> data/{split}.jsonl with the previous utterance of the dialogue, the clip path and the face-crop directory.

    Rows whose clip is missing or unreadable are dropped (MELD ships two such clips). Rows whose clip is too
    short or implausibly long, or that have no detected face, are kept with has_audio / has_vision = False,
    so the model learns to cope without that modality. `make faces` (python -m emo.faces) re-runs this at the end.
    """
    for split in SPLITS:
        rows = _read_csv(split)
        last_text: dict[int, str] = {}
        kept, dropped, no_audio, no_face = [], [], 0, 0
        for row in rows:  # rows are sorted, so "previous utterance" is well defined
            row["prev_text"] = last_text.get(row["dialogue_id"], "")
            last_text[row["dialogue_id"]] = row["text"]
            duration = _clip_duration(row["audio_path"])
            if duration is None:
                dropped.append(row["id"])
                continue
            plausible = MIN_AUDIO_SECONDS <= duration <= MAX_UTTERANCE_SECONDS
            row["duration_s"] = round(duration, 2)
            row["has_audio"] = plausible
            row["has_vision"] = plausible and any(FACES_DIR.joinpath(split, row["id"]).glob("*.jpg"))
            no_audio += not row["has_audio"]
            no_face += not row["has_vision"]
            kept.append(row)
        with open(DATA_DIR / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in kept)
        print(f"{split}: {len(kept)} utterances kept, {no_audio} without usable audio, {no_face} without a face, dropped {dropped or 'none'}")


def visual_input(row: dict) -> str | None:
    """What the robot would see for a manifest row: the raw clip when it was kept (live path), else the cached face crops."""
    if Path(row["video_path"]).exists():
        return row["video_path"]
    return row["faces_dir"] if row["has_vision"] else None


def load_manifest(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def _read_csv(split: str) -> list[dict]:
    with open(MELD_DIR / f"{split}.csv", encoding="utf-8") as f:
        rows = [
            {
                "id": f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}",
                "split": split,
                "dialogue_id": int(r["Dialogue_ID"]),
                "utterance_id": int(r["Utterance_ID"]),
                "speaker": r["Speaker"],
                "text": r["Utterance"].strip(),
                "emotion": r["Emotion"],
                "audio_path": _clip_path(split, r["Dialogue_ID"], r["Utterance_ID"]),
                "faces_dir": str(FACES_DIR / split / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"),
                "video_path": str(VIDEOS_DIR / split / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"),  # exists for a few test dialogues
            }
            for r in csv.DictReader(f)
        ]
    return sorted(rows, key=lambda r: (r["dialogue_id"], r["utterance_id"]))


def _clip_path(split: str, dialogue_id: str, utterance_id: str) -> str:
    """MELD re-extracted 132 test clips as final_videos_test<name>.flac; their plain-named twins are wrong clips."""
    name = f"dia{dialogue_id}_utt{utterance_id}.flac"
    fixed = MELD_DIR / "audio" / split / f"final_videos_test{name}"
    return str(fixed if fixed.exists() else MELD_DIR / "audio" / split / name)


def _clip_duration(path: str) -> float | None:
    try:
        return sf.info(path).duration
    except (RuntimeError, sf.LibsndfileError):
        return None


class MeldDataset(Dataset):
    """One item = (manifest row, {modality: cached features}, {modality: present?}) for the cached modalities given."""

    def __init__(self, rows: list[dict], features: dict[str, dict[str, torch.Tensor]]):
        self.rows = rows
        self.features = features  # e.g. {"vision": {id: [768]}, "audio": {id: [13, 768]}}; empty for text-only training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[dict, dict[str, torch.Tensor], dict[str, bool]]:
        row = self.rows[i]
        feats, present = {}, {}
        for modality, table in self.features.items():
            present[modality] = row[f"has_{modality}"] and row["id"] in table
            feats[modality] = table[row["id"]].float() if present[modality] else torch.zeros(FEATURE_SHAPES[modality])
        return row, feats, present


class Collate:
    """Turns dataset items into one batch: tokens (text + previous utterance), per-modality features and presence, labels."""

    def __init__(self, tokenizer, context: bool):
        self.tokenizer = tokenizer  # None when the model has no text branch
        self.context = context

    def __call__(self, items: list[tuple[dict, dict, dict]]) -> dict:
        rows, feats, present = zip(*items)
        batch = {"labels": torch.tensor([EMOTIONS.index(r["emotion"]) for r in rows])}
        for modality in feats[0]:
            batch[modality] = torch.stack([f[modality] for f in feats])
            batch[f"{modality}_present"] = torch.tensor([p[modality] for p in present])
        if self.tokenizer is not None:
            batch["tokens"] = tokenize(self.tokenizer, [r["text"] for r in rows], [r["prev_text"] for r in rows] if self.context else None)
        return batch


def tokenize(tokenizer, texts: list[str], prev_texts: list[str] | None) -> dict[str, torch.Tensor]:
    """Encodes each utterance, optionally paired with the previous utterance as a second segment."""
    return dict(tokenizer(texts, prev_texts, padding=True, truncation="longest_first", max_length=MAX_TEXT_TOKENS, return_tensors="pt"))


if __name__ == "__main__":
    download()
    build_manifests()
