#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc.face_engine import FaceEngine, capture_one_frame, choose_camera
from pc.paths import ENROLL_DIR, RAW_DIR, photo_reference, validate_user_id


def register_one_shot(camera: int | None = None, no_preview: bool = False) -> None:
    print("=== 一键注册人脸 (实时框框版) ===")

    cam_idx = choose_camera(preferred=camera)
    print(f"[info] 使用摄像头 {cam_idx}")

    frame = capture_one_frame(cam_idx, preview=not no_preview)
    if frame is None:
        raise RuntimeError("没有成功拍照")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = RAW_DIR / f"reg_{timestamp}.jpg"
    if not cv2.imwrite(str(photo_path), frame):
        raise RuntimeError("原始照片保存失败")
    print(f"[OK] 原始照片已保存: {photo_path}")

    engine = FaceEngine()
    face = engine.best_face(frame)
    if face is None:
        raise RuntimeError("未检测到人脸，请正对摄像头后重试")

    feat = engine.embed(frame, face)
    print(f"[OK] 特征维度: {feat.shape[0]}")

    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    default_id = f"user_{len(list(ENROLL_DIR.glob('*.json'))) + 1:03d}"
    user_id = input(f"请输入用户 ID (默认 {default_id}): ").strip() or default_id
    user_id = validate_user_id(user_id)

    emb_file = ENROLL_DIR / f"{user_id}.json"
    if emb_file.exists():
        raise RuntimeError(f"用户 ID 已存在: {user_id}")
    payload = {
        "id": user_id,
        "feature": feat.astype(float).tolist(),
        "dim": int(feat.shape[0]),
        "camera": cam_idx,
        "photo": photo_reference(photo_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with emb_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] 注册成功！用户 ID: {user_id}")
    print(f"[OK] 特征已保存: {emb_file}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="One-shot face enrollment")
    parser.add_argument("--camera", type=int, default=None, help="摄像头编号，例如 0/1/2")
    parser.add_argument("--no-preview", action="store_true", help="不弹预览窗，直接抓拍")
    args = parser.parse_args()

    try:
        register_one_shot(camera=args.camera, no_preview=args.no_preview)
        return 0
    except KeyboardInterrupt:
        print("\n[info] 用户中断")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
