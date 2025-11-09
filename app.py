from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2, numpy as np, base64
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join("veg", "models", "tomato.pt")
model = YOLO(model_path)

labels = {0: "Ripe", 1: "Turning", 2: "Unripe"}


@app.route("/")
def home():
    return "Tomato API working!"


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"].read()
    arr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.6)[0]

    if not results.boxes:
        return jsonify({"status": "Tomato not detected"})

    b = results.boxes[0]
    cls = int(b.cls)
    conf = float(b.conf)
    x1, y1, x2, y2 = map(int, b.xyxy[0])

    status = labels.get(cls, "Unknown")

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    return jsonify({
        "status": status,
        "confidence": round(conf, 2),
        "image_base64": b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

