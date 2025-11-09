from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_from_directory("veg", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image missing"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # ----------------------------------------------------------
    # STEP 1 — Detect Tomato Shape Using Canny + Contours
    # ----------------------------------------------------------
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    edges = cv2.Canny(blurred, 40, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tomato_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # remove small noise
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Roundish shapes = tomato
        if len(approx) > 6 and area > max_area:
            tomato_contour = cnt
            max_area = area

    if tomato_contour is None:
        return jsonify({"status": "Tomato not detected"})

    # Bounding box
    x, y, w, h = cv2.boundingRect(tomato_contour)
    crop = img[y:y+h, x:x+w]

    # ----------------------------------------------------------
    # STEP 2 — Ripeness Detection Using Dominant HSV Color
    # ----------------------------------------------------------
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Red
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = red1 + red2

    # Yellow/Orange
    yellow_mask = cv2.inRange(hsv, (10, 60, 50), (25, 255, 255))

    # Green
    green_mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))

    r = cv2.countNonZero(red_mask)
    yel = cv2.countNonZero(yellow_mask)
    g = cv2.countNonZero(green_mask)

    # ----------------------------------------------------------
    # DOMINANT RIPENESS = max color
    # ----------------------------------------------------------
    if r > yel and r > g:
        status = "Ripened"
        color = (0, 0, 255)
    elif yel > g:
        status = "Turning"
        color = (0, 255, 255)
    else:
        status = "Unripened"
        color = (0, 255, 0)

    # ----------------------------------------------------------
    # DRAW RESULT
    # ----------------------------------------------------------
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.putText(img, status, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    _, buffer = cv2.imencode(".jpg", img)
    encoded = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "status": status,
        "image_base64": encoded
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

