from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Serve the index.html (camera UI)
@app.route('/')
def serve_index():
    return send_from_directory(os.path.join(os.getcwd(), 'veg.py'), 'index.html')


# Tomato detection endpoint
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "Please upload an image with the field name 'image'"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color ranges
    ripened_lower = np.array([0, 150, 100])
    ripened_upper = np.array([10, 255, 255])
    turning_lower = np.array([10, 100, 100])
    turning_upper = np.array([25, 255, 255])
    unripened_lower = np.array([35, 50, 50])
    unripened_upper = np.array([85, 255, 255])

    # Masks
    mask_ripened = cv2.inRange(hsv, ripened_lower, ripened_upper)
    mask_turning = cv2.inRange(hsv, turning_lower, turning_upper)
    mask_unripened = cv2.inRange(hsv, unripened_lower, unripened_upper)

    ripened_count = cv2.countNonZero(mask_ripened)
    turning_count = cv2.countNonZero(mask_turning)
    unripened_count = cv2.countNonZero(mask_unripened)

    # Determine ripeness
    if ripened_count > turning_count and ripened_count > unripened_count:
        status = "Ripened"
    elif turning_count > unripened_count:
        status = "Turning"
    else:
        status = "Unripened"

    # Encode image (optional)
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
