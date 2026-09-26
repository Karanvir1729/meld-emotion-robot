"""Face-crop inputs: clip names in the MELD video archives (real, re-extracted, macOS stubs) and still images."""

import cv2
import numpy as np
import pytest

from emo import faces
from emo.faces import CLIP_NAME


def test_real_and_re_extracted_clips_match():
    assert CLIP_NAME.search("output_repeated_splits_test/dia12_utt3.mp4").groups() == (None, "dia12_utt3", "12")
    assert CLIP_NAME.search("output_repeated_splits_test/final_videos_testdia5_utt2.mp4").groups() == ("final_videos_test", "dia5_utt2", "5")


def test_macos_resource_fork_stubs_are_ignored():
    # The test archive holds a 212-byte "._" stub next to many clips; matching it once hid the real clip.
    assert CLIP_NAME.search("output_repeated_splits_test/._dia12_utt3.mp4") is None
    assert CLIP_NAME.search("output_repeated_splits_test/._final_videos_testdia5_utt2.mp4") is None


def test_a_still_image_is_one_frame(tmp_path, monkeypatch):
    path = tmp_path / "me.jpg"
    cv2.imwrite(str(path), np.full((120, 160, 3), 128, dtype=np.uint8))
    seen = []
    monkeypatch.setattr(faces, "largest_face", lambda frame: seen.append(frame.shape) or np.zeros((224, 224, 3), dtype=np.uint8))
    assert len(faces.faces_from_video(str(path))) == 1 and seen == [(120, 160, 3)]  # no ffmpeg, no detector download


def test_an_undecodable_video_raises(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        faces.faces_from_video(str(path))
