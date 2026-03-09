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
