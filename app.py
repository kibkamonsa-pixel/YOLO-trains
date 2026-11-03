
import os
import base64
import logging

import cv2
import numpy as np
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# URL เว็บไซต์ Netlify ที่อนุญาตให้เรียก API
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": FRONTEND_ORIGIN}})

# ตำแหน่งไฟล์โมเดล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RECTANGLE_MODEL_PATH = os.path.join(BASE_DIR, "best_rectangle.pt")
POLYGON_MODEL_PATH = os.path.join(BASE_DIR, "best_polygon.pt")

# เลือกอุปกรณ์ประมวลผล
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# โหลดโมเดล YOLO
rectangle_model = YOLO(RECTANGLE_MODEL_PATH, task="detect")
polygon_model = YOLO(POLYGON_MODEL_PATH, task="segment")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def image_to_base64(image):
    success, encoded = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("ไม่สามารถแปลงภาพได้")

    image_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + image_base64


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])

    return str(class_id)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Wildlife AI API is ready",
        "device": "GPU" if DEVICE == 0 else "CPU"
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "กรุณาเลือกภาพก่อนวิเคราะห์"
        }), 400

    file = request.files["image"]
    mode = request.form.get("mode", "rectangle").lower()

    if not file.filename:
        return jsonify({
            "error": "ไม่พบไฟล์ภาพ"
        }), 400

    extension = file.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "รองรับเฉพาะ JPG, JPEG, PNG และ WEBP"
        }), 400

    if mode not in ["rectangle", "polygon"]:
        return jsonify({
            "error": "กรุณาเลือก Rectangle หรือ Polygon"
        }), 400

    # อ่านภาพ
    image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if original is None:
        return jsonify({
            "error": "ไม่สามารถเปิดไฟล์ภาพได้"
        }), 400

    # เลือกโมเดล
    if mode == "rectangle":
        model = rectangle_model
    else:
        model = polygon_model

    # วิเคราะห์ภาพ
    results = model.predict(
        source=original,
        device=DEVICE,
        verbose=False
    )

    prediction = results[0]
    output = original.copy()

    counts = {}
    detections = []

    # เก็บข้อมูลการตรวจจับ
    if prediction.boxes is not None:

        for box in prediction.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item()) * 100

            name = get_class_name(model.names, class_id)

            counts[name] = counts.get(name, 0) + 1

            detections.append({
                "name": name,
                "confidence": round(confidence, 1)
            })

    # Rectangle
    if mode == "rectangle":

        output = prediction.plot()

    # Polygon
    else:

        # วาดเส้นขอบ Polygon สีน้ำเงิน ไม่เติมสีภายใน
        if prediction.masks is not None:

            for polygon in prediction.masks.xy:

                points = np.array(
                    polygon,
                    dtype=np.int32
                )

                if len(points) >= 3:
                    cv2.polylines(
                        output,
                        [points],
                        True,
                        (255, 0, 0),
                        3
                    )

        # เขียนข้อความสีน้ำเงิน
        if prediction.boxes is not None:

            for box in prediction.boxes:

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item()) * 100

                name = get_class_name(model.names, class_id)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                label = f"{name} {confidence:.1f}%"

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

    # ส่งผลลัพธ์กลับไปยัง Frontend
    return jsonify({

        "mode": (
            "Rectangle - Object Detection"
            if mode == "rectangle"
            else "Polygon - Instance Segmentation"
        ),

        "original": image_to_base64(original),
        "processed": image_to_base64(output),

        "total": len(detections),
        "counts": counts,
        "detections": detections,

        "device": "GPU" if DEVICE == 0 else "CPU"

    })


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "ขนาดภาพต้องไม่เกิน 12 MB"
    }), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
