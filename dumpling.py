from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import io
from ultralytics import YOLO
import hashlib
import json
import os

app = Flask(__name__)

# 加载 YOLO 模型
model = YOLO("model.pt")

# 设置静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# 设置缓存目录
CACHES_DIR = os.path.join(os.path.dirname(__file__), 'caches')

# 检测结果转成json输出
def detect_result_json(boxes):
    # 将检测结果转换为 JSON 格式
    results_json = boxes.data.tolist()  # 假设 boxes.data 是可以转换为列表的
    return json.dumps(results_json, indent=2)

# 图片检测函数
def detect(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # 计算图片的 hash 值
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_filename = f"{image_hash}.jpg"

    # 创建 caches 文件夹（如果不存在）
    if not os.path.exists(CACHES_DIR):
        os.makedirs(CACHES_DIR)

    # 构建图片的完整路径
    image_path = os.path.join(CACHES_DIR, image_filename)

    # 检查文件是否存在，如果不存在则保存图片
    if not os.path.exists(image_path):
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

    # 进行对象检测
    current_results = model.predict(source=image, conf=0.6, max_det=1000)
    result = current_results[0].boxes

    # 将检测结果转换为 JSON 格式
    results_json = detect_result_json(result)

    # 构建 JSON 文件的完整路径
    json_filename = f"{image_hash}.json"
    json_path = os.path.join(CACHES_DIR, json_filename)

    # 检查 JSON 文件是否存在，如果不存在则保存
    if not os.path.exists(json_path):
        with open(json_path, 'w') as json_file:
            json_file.write(results_json)

    # 返回结果BOXES
    return result

@app.route('/')
def web_index():
    return render_template('index.html', result=None)

@app.route('/detect', methods=['POST'])
def web_detect():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # 读取上传的图片
        image_bytes = file.read()

        # 检测结果
        boxes = detect(image_bytes)

        # 获取检测结果
        object_count = len(boxes.cls)
        object_info = f"识别到饺子数量：{object_count}个\n"

        # 返回结果
        return render_template('index.html', result=object_info)

# 提供 manifest.json 文件
@app.route('/manifest.json')
def web_serve_manifest():
    return send_from_directory(STATIC_DIR, 'manifest.json')

# 提供 icons 目录下的文件
@app.route('/icons/<path:filename>')
def web_serve_icon(filename):
    return send_from_directory(os.path.join(STATIC_DIR, 'icons'), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)