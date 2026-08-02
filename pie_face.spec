# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Pie Face desktop app.

Build a one-folder app (faster startup, easier debugging) that bundles the
OpenCV YuNet/SFace ONNX models and the data/ tree so end users get a fully
self-contained .app on macOS or .exe on Windows without touching a terminal.

Run from the project root:
    pyinstaller pie_face.spec
"""

from pathlib import Path

block_cipher = None

# Project root = directory containing this spec file.
ROOT = Path(SPECPATH)

datas = [
    # (source, dest_dir_inside_bundle)
    # User enrollment data is intentionally kept outside the read-only bundle.
    (str(ROOT / "models"), "models"),
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "pc.face_engine",
        "app.gui",
        "cv2",
        "numpy",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PyQt5.sip",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
        "pytest",
        "IPython",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PieFace",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PieFace",
)

# On macOS, bundle the collected folder into a .app package.
app = BUNDLE(
    coll,
    name="PieFace.app",
    icon=None,
    bundle_identifier="com.pie.face",
)
