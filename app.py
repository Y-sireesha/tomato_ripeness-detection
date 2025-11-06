from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64

from flask import Flask, request, jsonify, send_from_directory
import os


app = Flask(__name__)

# Ripeness Color Ranges in HSV
ripened_lower = np.array([0, 150, 100])
ripened_upper = np.array([10, 255, 255])
turning_lower = np.array([10, 100, 100])
turning_upper = np.array([25, 255, 255])
unripened_lower = np.array([35, 50, 50])
unripened_upper = np.array([85, 255, 255])

@app.route('/')
def home():
    return "✅ Tomato Ripeness Detection API is running! Use /detect endpoint with an image."

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Please upload an image with the field name "image"'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_ripened = cv2.inRange(hsv, ripened_lower, ripened_upper)
    mask_turning = cv2.inRange(hsv, turning_lower, turning_upper)
    mask_unripened = cv2.inRange(hsv, unripened_lower, unripened_upper)

    ripened_count = cv2.countNonZero(mask_ripened)
    turning_count = cv2.countNonZero(mask_turning)
    unripened_count = cv2.countNonZero(mask_unripened)

    if ripened_count > 5000:
        status = "Ripened"
    elif turning_count > 5000:
        status = "Turning"
    elif unripened_count > 5000:
        status = "Unripened"
    else:
        status = "Tomato not detected"

    # Encode frame with text overlay (optional)
    cv2.putText(frame, f"Status: {status}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': status,
        'ripened_pixels': int(ripened_count),
        'turning_pixels': int(turning_count),
        'unripened_pixels': int(unripened_count),
        'image_base64': img_base64
    })

@app.route('/index.html')
def serve_index():
    return send_from_directory(os.path.join(os.getcwd(), 'veg.py'), 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

