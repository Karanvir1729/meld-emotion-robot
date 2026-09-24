"""Manifest logic on a tiny synthetic split: previous-utterance context, sorting, dropped clips."""

import csv
import json

import numpy as np
import soundfile as sf

from emo import data
from emo.config import SAMPLE_RATE


def test_build_manifests_adds_context_and_drops_missing_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MELD_DIR", tmp_path / "meld")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "SPLITS", ["dev"])
    audio_dir = tmp_path / "meld" / "audio" / "dev"
    audio_dir.mkdir(parents=True)
    with open(tmp_path / "meld" / "dev.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"])
        writer.writerow([1, "Second line, with a comma", "Ross", "joy", "positive", 0, 1])  # out of order on purpose
        writer.writerow([2, "First line", "Monica", "neutral", "neutral", 0, 0])
        writer.writerow([3, "Other dialogue", "Joey", "anger", "negative", 1, 0])
        writer.writerow([4, "No clip on disk", "Joey", "fear", "negative", 1, 1])
    for name, seconds in (("dia0_utt0", 1.0), ("dia0_utt1", 0.1), ("dia1_utt0", 1.0), ("final_videos_testdia1_utt0", 2.0)):
        sf.write(audio_dir / f"{name}.flac", np.zeros(int(seconds * SAMPLE_RATE), dtype="float32"), SAMPLE_RATE)

    data.build_manifests()

    rows = [json.loads(line) for line in open(tmp_path / "dev.jsonl")]
    assert [r["id"] for r in rows] == ["dia0_utt0", "dia0_utt1", "dia1_utt0"]  # sorted, missing clip dropped
    assert [r["prev_text"] for r in rows] == ["", "First line", ""]  # context never crosses dialogues
    assert rows[1]["text"] == "Second line, with a comma"
    assert [r["has_audio"] for r in rows] == [True, False, True]  # 0.1 s clip is kept but flagged
    assert rows[2]["audio_path"].endswith("final_videos_testdia1_utt0.flac")  # MELD's re-extracted clip wins
