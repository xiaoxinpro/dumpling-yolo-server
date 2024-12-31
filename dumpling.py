from flask import Flask, request, render_template
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("model.pt")

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)