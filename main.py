#!/usr/bin/env python3
"""Entry point for the desktop app (source run and PyInstaller anchor)."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running `python main.py` from source.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
