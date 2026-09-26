"""load_audio: the dataset's 16 kHz FLAC is read directly; any other audio or video file goes through ffmpeg."""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from emo.audio import load_audio
from emo.config import MAX_AUDIO_SECONDS, SAMPLE_RATE


def test_16khz_flac_is_read_directly(tmp_path):
    path = tmp_path / "clip.flac"
    sf.write(path, np.full(2 * SAMPLE_RATE, 0.1, dtype="float32"), SAMPLE_RATE)
    assert len(load_audio(str(path))) == 2 * SAMPLE_RATE


def test_a_video_supplies_its_own_voice_resampled_and_cropped(tmp_path):
    # A 12 s clip with a 44.1 kHz tone and a picture, like a phone recording.
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=12",
                    "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=12", "-shortest", str(path)], check=True)
    wave = load_audio(str(path))
    assert len(wave) == int(MAX_AUDIO_SECONDS * SAMPLE_RATE)  # 16 kHz, first 10 s
    assert np.abs(wave).max() > 0.1  # the tone is there


def test_no_usable_audio_gives_none(tmp_path):
    path = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1", str(path)], check=True)
    assert load_audio(str(path)) is None  # no audio track
    short = tmp_path / "short.flac"
    sf.write(short, np.zeros(SAMPLE_RATE // 10, dtype="float32"), SAMPLE_RATE)
    assert load_audio(str(short)) is None  # 0.1 s is too short


def test_an_undecodable_file_raises(tmp_path):
    path = tmp_path / "notes.wav"
    path.write_text("this is not audio")
    with pytest.raises(ValueError):
        load_audio(str(path))
