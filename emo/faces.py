"""Face crops for the vision track: from a video file live, or from the streamed MELD archive for the dataset.

Torch-free on purpose: the extraction workers only need ffmpeg and OpenCV.
"""

import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from emo.config import FACE_DETECTOR, FACE_DETECTOR_URL, FACE_SIZE, FACES_DIR, FRAMES_PER_CLIP, KEPT_VIDEO_DIALOGUES, MAX_AUDIO_SECONDS, MELD_VIDEO_URL, VIDEOS_DIR

CLIP_NAME = re.compile(r"(?:^|/)(final_videos_test)?(dia(\d+)_utt\d+)\.mp4$")  # anchored: the test archive also holds macOS "._dia…" stubs
DETECT_WIDTH = 640  # frames are decoded at this width: faces stay well above the crop size, decoding is 4x cheaper
_detector = None  # one YuNet per process, created on first use


def video_frames(path: str, threads: int = 0) -> list[np.ndarray]:
    """Up to FRAMES_PER_CLIP frames (BGR, 640 px wide), evenly spaced over the first MAX_AUDIO_SECONDS of any ffmpeg-readable file.

    threads=0 lets ffmpeg use every core (one live turn); the dataset workers pass 1 because eight of them run at once.
    """
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-threads", str(threads), "-t", str(MAX_AUDIO_SECONDS), "-i", path,
                        "-an", "-vf", f"fps=2,scale={DETECT_WIDTH}:-2", f"{tmp}/%03d.jpg"], check=False)
        files = sorted(Path(tmp).glob("*.jpg"))
        picks = np.linspace(0, len(files) - 1, min(FRAMES_PER_CLIP, len(files))).round().astype(int) if files else []
        return [cv2.imread(str(files[i])) for i in picks]


def largest_face(frame: np.ndarray) -> np.ndarray | None:
    """The biggest detected face, cropped with a margin and resized to FACE_SIZE; None when no face is found."""
    global _detector
    if _detector is None:
        if not FACE_DETECTOR.exists():
            FACE_DETECTOR.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(FACE_DETECTOR_URL, FACE_DETECTOR)
        _detector = cv2.FaceDetectorYN.create(str(FACE_DETECTOR), "", (FACE_SIZE, FACE_SIZE), score_threshold=0.6)
    height, width = frame.shape[:2]
    _detector.setInputSize((width, height))
    _, faces = _detector.detect(frame)
    if faces is None:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
    margin = 0.2 * max(w, h)
    x0, y0 = int(max(0, x - margin)), int(max(0, y - margin))
    x1, y1 = int(min(width, x + w + margin)), int(min(height, y + h + margin))
    return cv2.resize(frame[y0:y1, x0:x1], (FACE_SIZE, FACE_SIZE))


def faces_from_video(path: str) -> list[Image.Image]:
    """Live path: a video or still image -> face crops as PIL images. The dataset is built with the same two functions."""
    image = cv2.imread(path)  # None for anything that is not a still image
    crops = [largest_face(frame) for frame in ([image] if image is not None else video_frames(path))]
    return [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops if c is not None]


def load_faces(faces_dir: str) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in sorted(Path(faces_dir).glob("*.jpg"))]


# --- dataset: stream the raw MELD videos once, keep only the face crops ----------

def save_faces(target: Path, mp4: bytes) -> int:
    """Worker: one clip's bytes -> face crops under target/. An empty directory records "no face found"."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(mp4)
        tmp.flush()
        crops = [largest_face(frame) for frame in video_frames(tmp.name, threads=1)]
    crops = [c for c in crops if c is not None]
    target.mkdir(parents=True, exist_ok=True)
    for k, crop in enumerate(crops):
        cv2.imwrite(str(target / f"{k}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(crops)


class RangedDownload(io.RawIOBase):
    """A large HTTP file as a forward-only stream, fetched by curl in byte ranges.

    Each range is a fresh connection (one long-lived connection got throttled, then reset), and a dropped
    range resumes at the byte where it stopped, so nothing is downloaded twice.
    """

    def __init__(self, url: str, chunk: int = 512 * 2**20, max_failures: int = 10):
        headers = subprocess.run(["curl", "-sSIL", url], capture_output=True, text=True, check=True).stdout.lower()
        self.size = int(re.findall(r"content-length: (\d+)", headers)[-1])  # of the final redirect target
        self.url, self.chunk, self.max_failures = url, chunk, max_failures
        self.offset, self.failures, self.curl = 0, 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while self.offset < self.size:
            if self.curl is None:
                end = min(self.offset + self.chunk, self.size) - 1
                self.curl = subprocess.Popen(["curl", "-sSfL", "-r", f"{self.offset}-{end}", self.url], stdout=subprocess.PIPE)
            n = self.curl.stdout.readinto(buffer)
            if n:
                self.offset, self.failures = self.offset + n, 0
                return n
            if self.curl.wait() != 0:  # the range dropped: the next one starts where it stopped
                self.failures += 1
                if self.failures > self.max_failures:
                    raise OSError(f"download of {self.url} keeps failing at byte {self.offset}")
            self.curl = None
        return 0


def download_faces() -> None:
    """Stream MELD.Raw.tar.gz (10.9 GB, never stored) into data/meld/faces/{split}/{id}/{k}.jpg,
    and keep the raw clips of KEPT_VIDEO_DIALOGUES (test) under data/meld/videos/.

    Clips that are already done are skipped, so an interrupted run can simply be started again.
    """
    largest_face(np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8))  # fetches the detector file before the workers start
    start = time.perf_counter()
    stream = io.BufferedReader(RangedDownload(MELD_VIDEO_URL), buffer_size=2**20)
    outer = tarfile.open(fileobj=stream, mode="r|gz")  # MELD.Raw/{train,dev,test}.tar.gz, one after the other
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for member in outer:
            if not (member.isfile() and member.name.endswith(".tar.gz")):
                continue
            split = Path(member.name).name.removesuffix(".tar.gz")
            inner = tarfile.open(fileobj=outer.extractfile(member), mode="r|gz")
            pending, clips, with_face = [], 0, 0
            for clip in inner:
                match = CLIP_NAME.search(clip.name)
                if not (clip.isfile() and match):
                    continue
                clips += 1
                name = match.group(2) + ("__fixed" if match.group(1) else "")
                target, video = FACES_DIR / split / name, VIDEOS_DIR / split / f"{name}.mp4"
                keep_video = split == "test" and int(match.group(3)) in KEPT_VIDEO_DIALOGUES
                if target.exists() and (video.exists() or not keep_video):
                    with_face += any(target.glob("*.jpg"))
                    continue
                mp4 = inner.extractfile(clip).read()
                if keep_video:
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(mp4)
                if target.exists():
                    with_face += any(target.glob("*.jpg"))
                    continue
                pending.append(pool.submit(save_faces, target, mp4))
                if len(pending) >= 2 * workers:  # bound memory: at most this many clips in flight
                    with_face += pending.pop(0).result() > 0
                if clips % 1000 == 0:
                    print(f"{split}: {clips} clips, {(time.perf_counter() - start) / 60:.0f} min", flush=True)
            with_face += sum(f.result() > 0 for f in pending)
            for fixed in [*(FACES_DIR / split).glob("*__fixed"), *(VIDEOS_DIR / split).glob("*__fixed.mp4")]:
                final = fixed.with_name(fixed.name.replace("__fixed", ""))  # MELD re-extracted some test clips; they replace their mis-cut twins
                shutil.rmtree(final, ignore_errors=True) if final.is_dir() else final.unlink(missing_ok=True)
                fixed.rename(final)
            print(f"{split}: {with_face}/{clips} clips with a face, {(time.perf_counter() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    download_faces()
    from emo.data import build_manifests

    build_manifests()  # picks up has_vision
