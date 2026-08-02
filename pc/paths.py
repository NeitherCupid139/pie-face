"""Shared resource and writable data paths for source and bundled runs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_data_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if getattr(sys, "frozen", False):
    # PyInstaller extracts bundled resources under _MEIPASS for onefile and
    # onedir builds. User data must stay outside that temporary/read-only tree.
    RESOURCE_ROOT = Path(
        getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
    )
    APP_DATA_ROOT = Path(user_data_dir("PieFace", appauthor=False))
else:
    RESOURCE_ROOT = PROJECT_ROOT
    APP_DATA_ROOT = Path(
        os.environ.get("PIE_FACE_DATA_DIR", str(PROJECT_ROOT))
    ).expanduser()

MODELS_DIR = RESOURCE_ROOT / "models"
DATA_DIR = APP_DATA_ROOT / "data"
ENROLL_DIR = DATA_DIR / "enrollments"
RAW_DIR = DATA_DIR / "raw" / "registered"


def validate_user_id(user_id: str) -> str:
    """Validate an ID before using it as an enrollment filename."""
    value = user_id.strip()
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if not value:
        raise ValueError("用户 ID 不能为空")
    if any(ch.isspace() for ch in value):
        raise ValueError("用户 ID 不能包含空格")
    if any(ch in value for ch in "/\\:") or value in {".", ".."}:
        raise ValueError("用户 ID 包含非法路径字符")
    if value.rstrip(" .") != value or value.casefold() in reserved:
        raise ValueError("用户 ID 不是有效的 Windows 文件名")
    return value


def photo_reference(photo_path: Path) -> str:
    """Return a portable path relative to the writable data directory."""
    return photo_path.relative_to(DATA_DIR).as_posix()
