#!/usr/bin/env python3
"""Download OpenCV Zoo face detection + recognition models."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, url: str, expected: str | None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        if expected and sha256_file(dest) != expected:
            print(f"[warn] checksum mismatch for {name}; downloading again")
            dest.unlink()
        else:
            print(f"[skip] {name} already exists")
            return dest

    print(f"[download] {name}")
    print(f"  from {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        if expected:
            got = sha256_file(tmp)
            if got != expected:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {name}: {got}")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[ok] {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> int:
    for name, meta in MODELS.items():
        download(name, meta["url"], meta["sha256"])
    print("All models ready under models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
