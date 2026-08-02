# Pie Face 桌面应用打包说明

## 项目结构

```text
pie/
├── app/gui.py                  # PyQt5 桌面 GUI
├── main.py                     # 应用入口
├── pc/paths.py                 # 模型和用户数据路径
├── pie_face.spec               # Mac/Windows onedir spec
├── pie_face_windows.spec       # Windows onefile spec
├── scripts/build_windows.ps1   # Windows 本地构建入口
├── requirements.txt            # 运行时和构建依赖
├── models/                     # 构建时下载的 ONNX 模型
└── data/                       # 源码运行时数据，不提交真实用户数据
```

## 源码运行

```bash
python -m pip install -r requirements.txt
python tools/download_models.py
python main.py
```

源码模式默认将注册数据写入项目内的 `data/`。设置 `PIE_FACE_DATA_DIR` 可以把它迁移到其他目录：

```bash
PIE_FACE_DATA_DIR=/path/to/pie-face-data python main.py
```

## Windows `.exe`

PyInstaller 不能从 macOS 交叉编译 Windows 程序，必须在 Windows 机器或 Windows 虚拟机上执行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\download_models.py
.\scripts\build_windows.ps1
```

构建产物：

```text
dist-win\PieFace.exe
```

该 exe 是单文件 GUI 程序，不打开控制台窗口。模型会被解包到 PyInstaller 的临时资源目录；注册特征和照片始终写入 `%LOCALAPPDATA%\PieFace\data`，不会写入 exe 所在目录。

也可以直接调用 PyInstaller：

```powershell
.\.venv\Scripts\python.exe -m PyInstaller `
  --noconfirm --clean `
  --distpath dist-win `
  --workpath build-win `
  pie_face_windows.spec
```

## macOS `.app`

在 macOS 上使用现有 onedir spec：

```bash
python -m PyInstaller pie_face.spec --noconfirm --clean
```

产物为 `dist/PieFace.app`。运行时数据写入 `~/Library/Application Support/PieFace/data`。未签名应用首次打开可能需要在 Finder 中右键选择“打开”，或执行：

```bash
xattr -cr dist/PieFace.app
```

## GitHub Actions Release

工作流文件为 `.github/workflows/windows-release.yml`。推送形如 `v0.1.0` 的标签后，GitHub Actions 会：

1. 在 Windows runner 安装 Python 3.11 和依赖。
2. 下载并校验 ONNX 模型。
3. 构建 `PieFace.exe`。
4. 创建 Release 并上传 `PieFace-windows-x64.exe` 与 `PieFace-windows-x64.zip`。

手动运行工作流只会生成 Actions artifact，不会创建 Release。

## 隐私与分发边界

`data/enrollments/` 中的 JSON 含人脸 embedding，`data/raw/registered/` 中的图片可能包含真实人脸。它们已被 `.gitignore` 排除，不应提交到公开仓库或通用 Release。发布前请检查：

```bash
git status --short
git check-ignore -v data/enrollments/*.json data/raw/registered/*.jpg
```

模型来自 OpenCV Zoo。再分发前请确认相关模型和第三方依赖的许可条款。Windows 用户如果遇到系统组件缺失，可能需要安装 Microsoft Visual C++ Redistributable。
