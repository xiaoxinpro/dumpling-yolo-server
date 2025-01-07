from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import io
from ultralytics import YOLO
import os

app = Flask(__name__)

# 加载 YOLO 模型
model = YOLO("model.pt")

# 设置静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # 读取上传的图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 进行对象检测
        current_results = model.predict(image)
        det_info = current_results[0].boxes.cls

        # 获取检测结果
        object_count = len(det_info)
        object_info = f"识别到饺子数量：{object_count}个\n"

        # 返回结果
        return render_template('index.html', result=object_info)

# 提供 manifest.json 文件
@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(STATIC_DIR, 'manifest.json')

# 提供 icons 目录下的文件
@app.route('/icons/<path:filename>')
def serve_icon(filename):
    return send_from_directory(os.path.join(STATIC_DIR, 'icons'), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)