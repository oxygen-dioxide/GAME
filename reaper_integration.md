# GAME Reaper脚本
本脚本基于GAME，将reaper中的音频区段识别为音符

## 安装
安装 Reaper 和 Python

下载 [GAME Onnx模型](https://github.com/openvpi/GAME/releases)，解压

在命令行中运行：

```cmd
pip install -r requirements_onnx.txt
pip install python-reapy
```

打开 Reaper ，在命令行中运行：

```cmd
python -c "import reapy; reapy.configure_reaper()"
```

重新启动 Reaper ，即可使用

## 使用
打开 Reaper ，选中需要转换的音频区段，在命令行中运行：

```cmd
python reaper_integration.py -m <模型文件夹>
```

# GAME Reaper Script

This script is based on **GAME** (Generative Adaptive MIDI Extractor) and is designed to identify audio items in Reaper as MIDI notes.

## Installation

1. **Install Reaper and Python** on your system.
2. **Download the [GAME Onnx Model](https://github.com/openvpi/GAME/releases/tag/v1.0.2)** and extract the files.
3. **Install dependencies** by running the following commands in your terminal:

```cmd
pip install -r requirements_onnx.txt
pip install python-reapy

```

4. **Configure Reaper integration** by running:

```cmd
python -c "import reapy; reapy.configure_reaper()"

```

5. **Restart Reaper** to finalize the setup.

---

## Usage

1. Open Reaper and **select the audio items** you wish to convert.
2. Run the following command in your terminal:

```cmd
python reaper_integration.py -m <path_to_model_folder>

```

