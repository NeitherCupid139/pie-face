# Pie Face

Pie Face 是一个基于 PyQt5、OpenCV YuNet 和 SFace 的桌面人脸注册与识别工具。

## 使用方式

1. 安装 Python 3.11 或更高版本。
2. 安装依赖：

   ```bash
   python -m pip install -r requirements.txt
   python tools/download_models.py
   ```

3. 启动桌面程序：

   ```bash
   python main.py
   ```

Windows 用户可以从 [Releases](https://github.com/CreedChung/pie-face/releases) 下载 `PieFace-windows-x64.exe`，无需安装 Python。首次运行时请在 Windows 设置中允许应用访问摄像头。

## 数据位置

模型是随程序打包的只读资源。注册特征和照片是本地运行数据，不会写入安装目录：

- Windows：`%LOCALAPPDATA%\PieFace\data`
- macOS：`~/Library/Application Support/PieFace/data`
- 源码运行：默认使用项目内 `data/`，也可以通过 `PIE_FACE_DATA_DIR` 指定数据根目录。

本公开仓库不包含真实人脸照片、注册特征或其他用户数据。请只在获得明确授权的情况下录入和处理人脸信息，并按适用法律管理数据的访问、保存和删除。

## Windows 打包

Windows `.exe` 必须在 Windows 环境构建，不能从 macOS 使用 PyInstaller 交叉编译：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\download_models.py
.\scripts\build_windows.ps1
```

产物为 `dist-win\PieFace.exe`。推送 `v*` 标签会触发 GitHub Actions，并自动创建带 exe 和 zip 的 Release。

## 模型来源

模型来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo)，下载脚本会校验 SHA-256。请在再分发前确认 OpenCV Zoo 及相关模型的许可条款。
