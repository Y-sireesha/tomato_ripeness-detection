from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__, static_folder=".")
CORS(app)  # allow frontend to call backend

# ----------------------------------------------------------
# SERVE INDEX.HTML
# ----------------------------------------------------------
@app.route("/")
def serve_home():
    return send_from_directory("veg", "index.html")

# ----------------------------------------------------------
# TOMATO RIPENESS DETECTION API
# ----------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ripened_lower = np.array([0, 150, 100])
    ripened_upper = np.array([10, 255, 255])
    turning_lower = np.array([10, 100, 100])
    turning_upper = np.array([25, 255, 255])
    unripened_lower = np.array([35, 50, 50])
    unripened_upper = np.array([85, 255, 255])

    mask_ripened = cv2.inRange(hsv, ripened_lower, ripened_upper)
    mask_turning = cv2.inRange(hsv, turning_lower, turning_upper)
    mask_unripened = cv2.inRange(hsv, unripened_lower, unripened_upper)

    r = cv2.countNonZero(mask_ripened)
    t = cv2.countNonZero(mask_turning)
    u = cv2.countNonZero(mask_unripened)

    if r > t and r > u:
        status = "Ripened"
    elif t > u:
        status = "Turning"
    else:
        status = "Unripened"

    _, buffer = cv2.imencode(".jpg", img)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "status": status,
        "ripened_pixels": int(r),
        "turning_pixels": int(t),
        "unripened_pixels": int(u),
        "image_base64": image_base64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

