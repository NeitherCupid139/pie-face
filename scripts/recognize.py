#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc.face_engine import (
    COLOR_FAIL,
    COLOR_INFO,
    COLOR_OK,
    COLOR_WARN,
    DEFAULT_MATCH_THRESHOLD,
    FaceEngine,
    capture_one_frame,
    choose_camera,
    draw_faces,
    draw_label,
    open_camera,
    warmup_camera,
)
from pc.paths import ENROLL_DIR


def load_enrollments() -> list[dict]:
    items: list[dict] = []
    for path in sorted(ENROLL_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "id" in data and "feature" in data:
            items.append(data)
    return items


def build_gallery(enrollments: list[dict]) -> list[tuple[str, np.ndarray]]:
    gallery: list[tuple[str, np.ndarray]] = []
    for item in enrollments:
        feat = np.asarray(item["feature"], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        gallery.append((str(item["id"]), feat))
    return gallery


def recognize_oneshot(
    camera: int | None = None,
    no_preview: bool = False,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> None:
    print("=== 人脸识别 (one-shot) ===")

    enrollments = load_enrollments()
    if not enrollments:
        raise RuntimeError("没有已注册用户，请先运行 scripts/register.py")

    cam_idx = choose_camera(preferred=camera)
    print(f"[info] 使用摄像头 {cam_idx}")
    frame = capture_one_frame(cam_idx, preview=not no_preview)

    engine = FaceEngine()
    face = engine.best_face(frame)
    if face is None:
        raise RuntimeError("未检测到人脸，请正对摄像头后重试")

    feat = engine.embed(frame, face)
    gallery = build_gallery(enrollments)
    best_id, best_score = engine.match_best(feat, gallery)

    print(f"[info] 候选用户数: {len(enrollments)}")
    print(f"[info] 最高相似度: {best_score:.4f} (阈值 {threshold})")
    if best_id is not None and best_score >= threshold:
        print(f"[OK] 识别成功: {best_id}")
    else:
        print("[FAIL] 未匹配到已注册用户")


def recognize_realtime(
    camera: int | None = None,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    process_every: int = 2,
) -> None:
    """Realtime face detection + recognition with live boxes."""
    print("=== 人脸识别 (realtime) ===")
    print("窗口内会实时画人脸框；匹配成功显示绿框+ID，未匹配显示红框。")
    print("操作：按 q/Esc 退出")

    enrollments = load_enrollments()
    if not enrollments:
        raise RuntimeError("没有已注册用户，请先运行 scripts/register.py")

    cam_idx = choose_camera(preferred=camera)
    print(f"[info] 使用摄像头 {cam_idx}")
    print(f"[info] 已加载用户: {', '.join(item['id'] for item in enrollments)}")
    print(f"[info] 匹配阈值: {threshold:.3f}")

    engine = FaceEngine()
    gallery = build_gallery(enrollments)

    cap = open_camera(cam_idx)
    window = f"recognize-realtime-{cam_idx}"
    process_every = max(1, int(process_every))

    # Cached recognition result so we don't re-embed every single frame.
    cached_labels: dict[int, str] = {}
    cached_colors: dict[int, tuple[int, int, int]] = {}
    last_match_text = "等待人脸..."
    last_match_color = COLOR_WARN
    frame_idx = 0
    fps = 0.0
    fps_t0 = time.time()
    fps_count = 0

    try:
        warmup_camera(cap)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame_idx += 1
            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                fps_t0 = now
                fps_count = 0

            faces = engine.detect(frame)
            labels: dict[int, str] = {}
            colors: dict[int, tuple[int, int, int]] = {}

            # Re-run recognition every N frames (detection stays every frame).
            if faces is not None and len(faces) > 0 and frame_idx % process_every == 0:
                for i, face in enumerate(faces):
                    try:
                        feat = engine.embed(frame, face)
                        best_id, best_score = engine.match_best(feat, gallery)
                    except Exception:
                        labels[i] = "embed-fail"
                        colors[i] = COLOR_FAIL
                        continue

                    if best_id is not None and best_score >= threshold:
                        labels[i] = f"{best_id} {best_score:.2f}"
                        colors[i] = COLOR_OK
                        last_match_text = f"MATCH: {best_id} ({best_score:.3f})"
                        last_match_color = COLOR_OK
                    else:
                        score_txt = f"{best_score:.2f}" if best_id is not None else "n/a"
                        labels[i] = f"unknown {score_txt}"
                        colors[i] = COLOR_FAIL
                        last_match_text = f"NO MATCH ({score_txt})"
                        last_match_color = COLOR_FAIL
                cached_labels = labels
                cached_colors = colors
            elif faces is not None and len(faces) > 0:
                # Reuse last recognition labels if face count still matches.
                if len(cached_labels) == len(faces):
                    labels = cached_labels
                    colors = cached_colors
                else:
                    for i in range(len(faces)):
                        labels[i] = "detecting..."
                        colors[i] = COLOR_WARN
            else:
                cached_labels = {}
                cached_colors = {}
                last_match_text = "等待人脸..."
                last_match_color = COLOR_WARN

            show = draw_faces(frame, faces, labels=labels, colors=colors)
            face_count = 0 if faces is None else len(faces)
            draw_label(
                show,
                f"faces: {face_count} | fps: {fps:.1f}",
                (12, 28),
                color=COLOR_INFO,
                scale=0.6,
                thickness=2,
            )
            draw_label(
                show,
                last_match_text,
                (12, 58),
                color=last_match_color,
                scale=0.6,
                thickness=2,
            )
            draw_label(
                show,
                f"thr={threshold:.3f} | Q quit",
                (12, 88),
                color=COLOR_INFO,
                scale=0.5,
            )

            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Face recognition (realtime / one-shot)")
    parser.add_argument("--camera", type=int, default=None, help="摄像头编号，例如 0/1/2")
    parser.add_argument(
        "--once",
        action="store_true",
        help="使用旧的 one-shot 模式（拍照后识别一次）",
    )
    parser.add_argument("--no-preview", action="store_true", help="one-shot 模式下不弹预览窗")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="匹配阈值",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="实时模式下每隔 N 帧做一次特征匹配（检测每帧都会做）",
    )
    args = parser.parse_args()

    try:
        if args.once:
            recognize_oneshot(
                camera=args.camera,
                no_preview=args.no_preview,
                threshold=args.threshold,
            )
        else:
            recognize_realtime(
                camera=args.camera,
                threshold=args.threshold,
                process_every=args.every,
            )
        return 0
    except KeyboardInterrupt:
        print("\n[info] 用户中断")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
