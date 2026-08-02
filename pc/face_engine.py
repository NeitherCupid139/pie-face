#!/usr/bin/env python3
"""Shared face detection + recognition helpers (OpenCV YuNet + SFace)."""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from pc.paths import MODELS_DIR

DET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
REC_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine similarity threshold for SFace (OpenCV docs recommend ~0.363)
DEFAULT_MATCH_THRESHOLD = 0.363

# BGR colors
COLOR_OK = (40, 200, 40)
COLOR_WARN = (0, 200, 255)
COLOR_FAIL = (40, 40, 220)
COLOR_INFO = (230, 230, 230)
COLOR_BOX = (0, 220, 255)


class FaceEngine:
    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        if not DET_MODEL.exists():
            raise FileNotFoundError(f"缺少检测模型: {DET_MODEL}")
        if not REC_MODEL.exists():
            raise FileNotFoundError(f"缺少识别模型: {REC_MODEL}")

        self.detector = cv2.FaceDetectorYN.create(
            str(DET_MODEL),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")
        self._last_input_size: tuple[int, int] | None = None

    def detect(self, image_bgr: np.ndarray) -> np.ndarray | None:
        h, w = image_bgr.shape[:2]
        size = (w, h)
        if self._last_input_size != size:
            self.detector.setInputSize(size)
            self._last_input_size = size
        _, faces = self.detector.detect(image_bgr)
        return faces

    def best_face(self, image_bgr: np.ndarray) -> np.ndarray | None:
        faces = self.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None
        # faces rows: x, y, w, h, landmarks..., score
        scores = faces[:, -1]
        return faces[int(np.argmax(scores))]

    def embed(self, image_bgr: np.ndarray, face: np.ndarray | None = None) -> np.ndarray:
        if face is None:
            face = self.best_face(image_bgr)
        if face is None:
            raise ValueError("未检测到人脸")
        aligned = self.recognizer.alignCrop(image_bgr, face)
        feat = self.recognizer.feature(aligned)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        # L2 normalize for stable cosine matching
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def match_score(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        a = np.asarray(feat_a, dtype=np.float32).reshape(-1)
        b = np.asarray(feat_b, dtype=np.float32).reshape(-1)
        # Features are already L2-normalized in embed(), but keep safe division.
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def match_best(
        self,
        feat: np.ndarray,
        gallery: list[tuple[str, np.ndarray]],
    ) -> tuple[str | None, float]:
        """Return (best_id, best_score) against an enrollment gallery."""
        best_id: str | None = None
        best_score = -1.0
        query = np.asarray(feat, dtype=np.float32).reshape(-1)
        for user_id, enrolled in gallery:
            score = self.match_score(query, enrolled)
            if score > best_score:
                best_score = score
                best_id = user_id
        return best_id, best_score


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = COLOR_INFO,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    top = max(0, y - th - pad * 2)
    left = max(0, x)
    cv2.rectangle(
        image,
        (left, top),
        (left + tw + pad * 2, top + th + baseline + pad * 2),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        text,
        (left + pad, top + th + pad),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_faces(
    image: np.ndarray,
    faces: np.ndarray | None,
    labels: dict[int, str] | None = None,
    colors: dict[int, tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = COLOR_BOX,
) -> np.ndarray:
    """Draw face boxes (+ optional labels) onto a copy of the image."""
    out = image.copy()
    if faces is None or len(faces) == 0:
        return out

    labels = labels or {}
    colors = colors or {}
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(int)
        color = colors.get(i, default_color)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # 5 facial landmarks if present
        if face.shape[0] >= 15:
            for j in range(5):
                lx = int(face[4 + j * 2])
                ly = int(face[5 + j * 2])
                cv2.circle(out, (lx, ly), 2, color, -1)

        score = float(face[-1])
        label = labels.get(i)
        if label:
            text = f"{label} | {score:.2f}"
        else:
            text = f"face {score:.2f}"
        draw_label(out, text, (x, y - 6), color=color)
    return out


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}")
    # Prefer a moderate resolution for smoother realtime detection.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def warmup_camera(cap: cv2.VideoCapture, frames: int = 8) -> np.ndarray:
    frame = None
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            frame = None
    if frame is None:
        raise RuntimeError("摄像头读帧失败")
    return frame


def list_cameras(max_index: int = 8) -> list[dict[str, Any]]:
    """Probe available cameras and return basic info."""
    found: list[dict[str, Any]] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
        cap.release()
        if not ok or frame is None:
            continue
        found.append(
            {
                "index": idx,
                "width": width,
                "height": height,
                "backend": backend,
            }
        )
    return found


def choose_camera(preferred: int | None = None) -> int:
    cameras = list_cameras()
    if not cameras:
        raise RuntimeError("未找到可用摄像头")

    if preferred is not None:
        if any(c["index"] == preferred for c in cameras):
            return preferred
        print(f"[warn] 指定摄像头 {preferred} 不可用，改为手动选择")

    print("可用摄像头：")
    for cam in cameras:
        print(
            f"  [{cam['index']}] {cam['width']}x{cam['height']}  backend={cam['backend']}"
        )

    default_idx = cameras[0]["index"]
    raw = input(f"请选择摄像头编号 (默认 {default_idx}): ").strip()
    if not raw:
        return default_idx
    try:
        selected = int(raw)
    except ValueError as exc:
        raise RuntimeError("摄像头编号必须是整数") from exc
    if not any(c["index"] == selected for c in cameras):
        raise RuntimeError(f"摄像头 {selected} 不在可用列表中")
    return selected


def capture_one_frame(
    camera_index: int,
    preview: bool = True,
    engine: FaceEngine | None = None,
) -> np.ndarray:
    """Open camera, optionally show realtime face boxes, then capture one frame."""
    cap = open_camera(camera_index)
    try:
        frame = warmup_camera(cap)

        if not preview:
            print("正在拍照...")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头读帧失败")
            return frame

        if engine is None:
            engine = FaceEngine()

        print("预览窗口已打开：实时检测人脸框 | 空格/Enter 拍照 | q/Esc 取消")
        window = f"camera-{camera_index}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            ok, live = cap.read()
            if not ok or live is None:
                continue

            faces = engine.detect(live)
            show = draw_faces(live, faces, default_color=COLOR_BOX)

            face_count = 0 if faces is None else len(faces)
            status = f"faces: {face_count}"
            status_color = COLOR_OK if face_count > 0 else COLOR_WARN
            draw_label(show, status, (12, 28), color=status_color, scale=0.65, thickness=2)
            draw_label(
                show,
                f"Cam {camera_index} | SPACE capture | Q cancel",
                (12, 58),
                color=COLOR_INFO,
                scale=0.55,
            )

            cv2.imshow(window, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 32):  # Enter / Space
                frame = live
                break
            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError("用户取消拍照")
        return frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_camera_loop(
    camera_index: int,
    on_frame: Callable[[np.ndarray, FaceEngine], np.ndarray | None],
    *,
    engine: FaceEngine | None = None,
    window_name: str = "realtime",
    quit_keys: tuple[int, ...] = (27, ord("q"), ord("Q")),
) -> None:
    """Generic realtime camera loop. `on_frame` returns the image to display."""
    if engine is None:
        engine = FaceEngine()

    cap = open_camera(camera_index)
    try:
        warmup_camera(cap)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"实时窗口已打开：按 q/Esc 退出 ({window_name})")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            display = on_frame(frame, engine)
            if display is None:
                display = frame

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in quit_keys:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
