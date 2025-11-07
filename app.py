from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)  # allows camera web page to access API

# ✅ Serve the camera HTML page
@app.route('/')
def serve_index():
    return send_from_directory(os.path.join(os.getcwd(), 'veg.py'), 'index.html')


# ✅ Tomato ripeness detection API
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Please upload an image with the field name 'image'"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color thresholds (tune as needed)
    ripened_lower = np.array([0, 150, 100])     # red
    ripened_upper = np.array([10, 255, 255])
    turning_lower = np.array([10, 100, 100])    # yellow/orange
    turning_upper = np.array([25, 255, 255])
    unripened_lower = np.array([35, 50, 50])    # green
    unripened_upper = np.array([85, 255, 255])

    # Create color masks
    mask_ripened = cv2.inRange(hsv, ripened_lower, ripened_upper)
    mask_turning = cv2.inRange(hsv, turning_lower, turning_upper)
    mask_unripened = cv2.inRange(hsv, unripened_lower, unripened_upper)

    # Count pixels in each range
    ripened_count = cv2.countNonZero(mask_ripened)
    turning_count = cv2.countNonZero(mask_turning)
    unripened_count = cv2.countNonZero(mask_unripened)

    # Decide ripeness
    if ripened_count > turning_count and ripened_count > unripened_count:
        status = "🍅 Ripened"
        color = (0, 0, 255)
    elif turning_count > unripened_count:
        status = "🟡 Turning"
        color = (0, 255, 255)
    else:
        status = "🟢 Unripened"
        color = (0, 255, 0)

    # Draw status text on image
    cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Encode result image
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": status,
        "ripened_pixels": int(ripened_count),
        "turning_pixels": int(turning_count),
        "unripened_pixels": int(unripened_count),
        "image_base64": image_base64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
