# 饺子数量检测服务

## 简介

该项目是一个基于Flask和YOLO模型的Web应用，用于检测上传图片中的饺子数量。用户可以通过网页界面上传一张包含饺子的图片，系统将自动识别并返回饺子的数量。

> 开源地址：https://github.com/xiaoxinpro/dumpling-yolo-server

## 功能特点

- 用户可以上传任意格式的图片文件。
- 使用YOLO模型进行高效的物体检测。
- 实时显示检测结果，并提供友好的用户交互体验。


## 项目结构
```
dumpling-yolo-server 
 ├─templates      # 前端页面模板
 │  └─index.html  # 首页模板
 ├─ultralytics    # YOLO模型库
 ├─dumpling.py    # 后端主程序
 ├─model.pt       # YOLO预训练模型文件 
 └─README.md      # 项目说明文档
```

## 安装与运行

### 环境准备

确保已安装以下依赖项：
- [Python 3.9+](https://www.python.org/downloads/)
- [pip3](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-you-can-run-pip-from-the-command-line)
- [pytorch](https://pytorch.org/get-started/locally/)

### 安装依赖

在项目根目录下执行以下命令以安装所需的Python库：

```bash
pip3 install flask opencv-python-headless matplotlib pyyaml tqdm requests psutil
```

### 运行应用
启动Flask开发服务器：

```bash
python dumpling.py
```

默认情况下，应用将在 http://localhost:5000/ 上运行，打开浏览器访问该地址即可使用服务。

## 使用方法
访问首页，点击“选择文件”按钮上传一张包含饺子的图片。
点击“检测饺子”按钮，系统将开始处理图片并显示检测进度。
检测完成后，页面会显示识别到的饺子数量。

## 注意事项
请确保上传的图片中包含清晰可见的饺子，以便获得更准确的检测结果。
如果遇到任何问题或错误，请检查控制台输出日志，或通过GitHub提交Issue寻求帮助。
