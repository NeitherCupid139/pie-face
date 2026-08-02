#!/usr/bin/env python3
"""PyQt5 desktop GUI for face enrollment and recognition.

Replaces the terminal scripts (scripts/register.py, scripts/recognize.py)
with a point-and-click window. Bundled via PyInstaller into a Mac .app
and a Windows .exe so end users never touch a terminal.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Keep the source root importable for direct script execution. Runtime resource
# and writable user data locations are centralized in pc.paths.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pc.face_engine import (
    COLOR_BOX,
    COLOR_FAIL,
    COLOR_INFO,
    COLOR_OK,
    COLOR_WARN,
    DEFAULT_MATCH_THRESHOLD,
    FaceEngine,
    draw_faces,
    draw_label,
    list_cameras,
    open_camera,
    warmup_camera,
)
from pc.paths import (
    DATA_DIR,
    ENROLL_DIR,
    RAW_DIR,
    RESOURCE_ROOT,
    photo_reference,
    validate_user_id,
)


def log_to_widget(widget: QTextEdit, msg: str) -> None:
    widget.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


class CameraThread(QThread):
    """Background thread streaming the camera into a QLabel.

    Emits `frame_ready(QPixmap)` for each rendered frame and
    `finished_stream()` when the loop ends.
    """

    frame_ready = pyqtSignal(QPixmap)
    status_ready = pyqtSignal(str, tuple)
    finished_stream = pyqtSignal()

    def __init__(
        self,
        camera_index: int,
        mode: str = "detect",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        process_every: int = 2,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.mode = mode  # "detect" | "recognize"
        self.threshold = threshold
        self.process_every = max(1, int(process_every))
        self._running = True
        self._engine: Optional[FaceEngine] = None
        self._gallery: list[tuple[str, np.ndarray]] = []

    def stop(self) -> None:
        self._running = False

    def _load_gallery(self) -> None:
        self._gallery = []
        for path in sorted(ENROLL_DIR.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "id" in data and "feature" in data:
                feat = np.asarray(data["feature"], dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                self._gallery.append((str(data["id"]), feat))

    def run(self) -> None:
        try:
            cap = open_camera(self.camera_index)
        except Exception as exc:
            self.status_ready.emit(f"摄像头打开失败: {exc}", (COLOR_FAIL,))
            self.finished_stream.emit()
            return

        try:
            warmup_camera(cap)
            if self._engine is None:
                self._engine = FaceEngine()
            if self.mode == "recognize":
                self._load_gallery()

            frame_idx = 0
            cached_labels: dict[int, str] = {}
            cached_colors: dict[int, tuple[int, int, int]] = {}
            last_match_text = "等待人脸..."
            last_match_color = COLOR_WARN
            fps = 0.0
            fps_t0 = time.time()
            fps_count = 0

            while self._running:
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

                faces = self._engine.detect(frame)
                labels: dict[int, str] = {}
                colors: dict[int, tuple[int, int, int]] = {}

                if self.mode == "recognize" and faces is not None and len(faces) > 0:
                    if frame_idx % self.process_every == 0:
                        for i, face in enumerate(faces):
                            try:
                                feat = self._engine.embed(frame, face)
                                best_id, best_score = self._engine.match_best(
                                    feat, self._gallery
                                )
                            except Exception:
                                labels[i] = "embed-fail"
                                colors[i] = COLOR_FAIL
                                continue
                            if best_id is not None and best_score >= self.threshold:
                                labels[i] = f"{best_id} {best_score:.2f}"
                                colors[i] = COLOR_OK
                                last_match_text = f"MATCH: {best_id} ({best_score:.3f})"
                                last_match_color = COLOR_OK
                            else:
                                score_txt = (
                                    f"{best_score:.2f}"
                                    if best_id is not None
                                    else "n/a"
                                )
                                labels[i] = f"unknown {score_txt}"
                                colors[i] = COLOR_FAIL
                                last_match_text = f"NO MATCH ({score_txt})"
                                last_match_color = COLOR_FAIL
                        cached_labels = labels
                        cached_colors = colors
                    else:
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
                    if self.mode == "recognize":
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
                if self.mode == "recognize":
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
                        f"thr={self.threshold:.3f} | {len(self._gallery)} users",
                        (12, 88),
                        color=COLOR_INFO,
                        scale=0.5,
                    )
                else:
                    draw_label(
                        show,
                        f"Cam {self.camera_index} | detect mode",
                        (12, 58),
                        color=COLOR_INFO,
                        scale=0.55,
                    )

                pixmap = _frame_to_pixmap(show)
                if pixmap is not None:
                    self.frame_ready.emit(pixmap)

            self.finished_stream.emit()
        except Exception as exc:
            self.status_ready.emit(f"摄像头线程异常: {exc}", (COLOR_FAIL,))
            self.finished_stream.emit()
        finally:
            try:
                cap.release()
            except Exception:
                pass


class CaptureThread(QThread):
    """Open camera, show live preview with face boxes, capture on demand."""

    frame_ready = pyqtSignal(QPixmap)
    captured = pyqtSignal(np.ndarray)
    failed = pyqtSignal(str)
    finished_stream = pyqtSignal()

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self.camera_index = camera_index
        self._running = True
        self._capture_next = False
        self._engine: Optional[FaceEngine] = None

    def stop(self) -> None:
        self._running = False

    def capture_now(self) -> None:
        self._capture_next = True

    def run(self) -> None:
        try:
            cap = open_camera(self.camera_index)
        except Exception as exc:
            self.failed.emit(f"摄像头打开失败: {exc}")
            return

        try:
            warmup_camera(cap)
            if self._engine is None:
                self._engine = FaceEngine()

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                if self._capture_next:
                    self.captured.emit(frame.copy())
                    self._capture_next = False
                    continue

                faces = self._engine.detect(frame)
                show = draw_faces(frame, faces, default_color=COLOR_BOX)
                face_count = 0 if faces is None else len(faces)
                status = f"faces: {face_count}"
                status_color = COLOR_OK if face_count > 0 else COLOR_WARN
                draw_label(show, status, (12, 28), color=status_color, scale=0.65, thickness=2)
                draw_label(
                    show,
                    f"Cam {self.camera_index} | click Capture to enroll",
                    (12, 58),
                    color=COLOR_INFO,
                    scale=0.55,
                )
                pixmap = _frame_to_pixmap(show)
                if pixmap is not None:
                    self.frame_ready.emit(pixmap)

            self.finished_stream.emit()
        except Exception as exc:
            self.failed.emit(f"预览线程异常: {exc}")
        finally:
            try:
                cap.release()
            except Exception:
                pass


def _frame_to_pixmap(frame: np.ndarray) -> Optional[QPixmap]:
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("人脸识别系统 - Pie Face")
        self.resize(1080, 760)

        self.engine = FaceEngine()
        self.camera_thread: Optional[CameraThread] = None
        self.capture_thread: Optional[CaptureThread] = None
        self._pending_capture: Optional[np.ndarray] = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # --- top control row ---
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(220)
        top_row.addWidget(self.camera_combo)
        self.refresh_cam_btn = QPushButton("刷新摄像头")
        self.refresh_cam_btn.clicked.connect(self.refresh_cameras)
        top_row.addWidget(self.refresh_cam_btn)
        top_row.addStretch()

        self.start_rec_btn = QPushButton("▶ 开始实时识别")
        self.start_rec_btn.clicked.connect(self.start_recognize)
        top_row.addWidget(self.start_rec_btn)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.clicked.connect(self.stop_all_streams)
        self.stop_btn.setEnabled(False)
        top_row.addWidget(self.stop_btn)
        root_layout.addLayout(top_row)

        # --- params row ---
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("匹配阈值:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(DEFAULT_MATCH_THRESHOLD)
        param_row.addWidget(self.threshold_spin)
        param_row.addWidget(QLabel("识别间隔(帧):"))
        self.every_spin = QSpinBox()
        self.every_spin.setRange(1, 30)
        self.every_spin.setValue(2)
        param_row.addWidget(self.every_spin)
        param_row.addStretch()
        root_layout.addLayout(param_row)

        # --- main body: video + side panel ---
        body_row = QHBoxLayout()

        self.video_label = QLabel("摄像头预览区")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888; font-size: 16px;"
        )
        body_row.addWidget(self.video_label, stretch=3)

        side_panel = QVBoxLayout()
        side_panel.addWidget(QLabel("已注册用户:"))
        self.user_list = QListWidget()
        self.user_list.setMinimumWidth(240)
        side_panel.addWidget(self.user_list)

        refresh_users_btn = QPushButton("刷新用户列表")
        refresh_users_btn.clicked.connect(self.refresh_users)
        side_panel.addWidget(refresh_users_btn)

        side_panel.addSpacing(12)
        side_panel.addWidget(QLabel("注册新用户:"))
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("用户 ID (留空自动编号)")
        side_panel.addWidget(self.user_id_input)

        self.register_btn = QPushButton("📷 打开摄像头注册")
        self.register_btn.clicked.connect(self.start_register_preview)
        side_panel.addWidget(self.register_btn)

        self.capture_btn = QPushButton("✓ 拍照并注册")
        self.capture_btn.clicked.connect(self.do_register_capture)
        self.capture_btn.setEnabled(False)
        side_panel.addWidget(self.capture_btn)

        body_row.addLayout(side_panel, stretch=1)
        root_layout.addLayout(body_row, stretch=1)

        # --- log area ---
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("运行日志"))
        root_row.addStretch()
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(lambda: self.log_area.clear())
        root_row.addWidget(clear_btn)
        root_layout.addLayout(root_row)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(140)
        root_layout.addWidget(self.log_area)

        self.setStatusBar(QStatusBar())
        self.refresh_cameras()
        self.refresh_users()
        log_to_widget(self.log_area, f"应用启动 | 模型目录: {RESOURCE_ROOT / 'models'}")

    # ---------- camera helpers ----------
    def refresh_cameras(self) -> None:
        self.camera_combo.clear()
        try:
            cameras = list_cameras()
        except Exception as exc:
            self.statusBar().showMessage(f"摄像头列表失败: {exc}")
            return
        if not cameras:
            self.camera_combo.addItem("（未找到摄像头）", userData=-1)
            self.statusBar().showMessage("未找到可用摄像头")
            return
        for cam in cameras:
            label = f"[{cam['index']}] {cam['width']}x{cam['height']} ({cam['backend']})"
            self.camera_combo.addItem(label, userData=cam["index"])
        self.statusBar().showMessage(f"已发现 {len(cameras)} 个摄像头")

    def current_camera_index(self) -> Optional[int]:
        idx = self.camera_combo.currentData()
        return None if idx is None or idx < 0 else int(idx)

    def refresh_users(self) -> None:
        self.user_list.clear()
        if not ENROLL_DIR.exists():
            self.user_list.addItem("（暂无注册用户）")
            return
        files = sorted(ENROLL_DIR.glob("*.json"))
        if not files:
            self.user_list.addItem("（暂无注册用户）")
            return
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.user_list.addItem(f"{data.get('id', path.stem)}")
            except Exception:
                self.user_list.addItem(path.stem)

    # ---------- streaming controls ----------
    def _set_streaming_ui(self, streaming: bool) -> None:
        self.start_rec_btn.setEnabled(not streaming)
        self.register_btn.setEnabled(not streaming)
        self.stop_btn.setEnabled(streaming)
        self.refresh_cam_btn.setEnabled(not streaming)
        self.capture_btn.setEnabled(False)

    def start_recognize(self) -> None:
        cam_idx = self.current_camera_index()
        if cam_idx is None:
            QMessageBox.warning(self, "摄像头", "请先选择一个摄像头")
            return
        if not ENROLL_DIR.exists() or not list(ENROLL_DIR.glob("*.json")):
            QMessageBox.warning(
                self,
                "无注册用户",
                "没有任何已注册用户，请先在右侧注册。",
            )
            return
        self.stop_all_streams()
        self.camera_thread = CameraThread(
            camera_index=cam_idx,
            mode="recognize",
            threshold=self.threshold_spin.value(),
            process_every=self.every_spin.value(),
        )
        self.camera_thread.frame_ready.connect(self._show_pixmap)
        self.camera_thread.finished_stream.connect(
            lambda: self._set_streaming_ui(False)
        )
        self.camera_thread.status_ready.connect(self._on_status)
        self.camera_thread.start()
        self._set_streaming_ui(True)
        log_to_widget(
            self.log_area,
            f"实时识别已启动 | cam={cam_idx} thr={self.threshold_spin.value():.3f}",
        )

    def start_register_preview(self) -> None:
        cam_idx = self.current_camera_index()
        if cam_idx is None:
            QMessageBox.warning(self, "摄像头", "请先选择一个摄像头")
            return
        self.stop_all_streams()
        self.capture_thread = CaptureThread(camera_index=cam_idx)
        self.capture_thread.frame_ready.connect(self._show_pixmap)
        self.capture_thread.captured.connect(self._on_captured)
        self.capture_thread.failed.connect(self._on_failed)
        self.capture_thread.finished_stream.connect(
            lambda: self._set_streaming_ui(False)
        )
        self.capture_thread.start()
        self._set_streaming_ui(True)
        self.capture_btn.setEnabled(True)
        log_to_widget(self.log_area, f"注册预览已启动 | cam={cam_idx}")

    def do_register_capture(self) -> None:
        if self.capture_thread is not None:
            self.capture_thread.capture_now()

    def _on_captured(self, frame: np.ndarray) -> None:
        self._pending_capture = frame
        log_to_widget(self.log_area, "已抓拍一帧，正在提取特征...")
        try:
            face = self.engine.best_face(frame)
            if face is None:
                log_to_widget(self.log_area, "❌ 未检测到人脸，请正对摄像头重试")
                QMessageBox.warning(self, "未检测到人脸", "请正对摄像头后重试")
                return
            feat = self.engine.embed(frame, face)
            user_id = self.user_id_input.text().strip()
            ENROLL_DIR.mkdir(parents=True, exist_ok=True)
            if not user_id:
                serial = len(list(ENROLL_DIR.glob("*.json"))) + 1
                user_id = f"user_{serial:03d}"
                while (ENROLL_DIR / f"{user_id}.json").exists():
                    serial += 1
                    user_id = f"user_{serial:03d}"
            user_id = validate_user_id(user_id)
            emb_file = ENROLL_DIR / f"{user_id}.json"
            if emb_file.exists():
                raise RuntimeError(f"用户 ID 已存在: {user_id}")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_path = RAW_DIR / f"reg_{timestamp}.jpg"
            if not cv2.imwrite(str(photo_path), frame):
                raise RuntimeError("原始照片保存失败")
            payload = {
                "id": user_id,
                "feature": feat.astype(float).tolist(),
                "dim": int(feat.shape[0]),
                "camera": -1,
                "photo": photo_reference(photo_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with emb_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log_to_widget(self.log_area, f"✅ 注册成功: {user_id} | 特征维度: {feat.shape[0]}")
            self.user_id_input.clear()
            self.refresh_users()
        except Exception as exc:
            log_to_widget(self.log_area, f"❌ 注册失败: {exc}")
            QMessageBox.critical(self, "注册失败", str(exc))

    def _on_failed(self, msg: str) -> None:
        log_to_widget(self.log_area, f"❌ {msg}")
        self._set_streaming_ui(False)

    def _on_status(self, msg: str, color: tuple) -> None:
        log_to_widget(self.log_area, msg)

    def _show_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def stop_all_streams(self) -> None:
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait(2000)
        self.camera_thread = None
        if self.capture_thread is not None and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread.wait(2000)
        self.capture_thread = None
        self.capture_btn.setEnabled(False)

    def closeEvent(self, event) -> None:
        self.stop_all_streams()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Pie Face")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
