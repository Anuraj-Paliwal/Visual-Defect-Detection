import os
import cv2
import uuid
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CROP_DIR   = os.path.join(BASE_DIR, "crops")
# ROOT FIX: Using 'Small' model instead of 'Nano' for better feature extraction
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s.pt") 

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR,   exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'detections.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "change-me-in-production"

db = SQLAlchemy(app)

# ── Model ─────────────────────────────────────────────────────────────────────
class Detection(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    source      = db.Column(db.String(64))
    class_name  = db.Column(db.String(64))
    confidence  = db.Column(db.Float)
    bbox_x1     = db.Column(db.Integer)
    bbox_y1     = db.Column(db.Integer)
    bbox_x2     = db.Column(db.Integer)
    bbox_y2     = db.Column(db.Integer)
    crop_path   = db.Column(db.String(256))
    frame_path  = db.Column(db.String(256))

    def to_dict(self):
        return {
            "id":          self.id,
            "timestamp":   self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source":      self.source,
            "class_name":  self.class_name,
            "confidence":  round(self.confidence, 3),
            "bbox":        [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "crop_url":    f"/crops/{os.path.basename(self.crop_path)}" if self.crop_path else None,
            "frame_url":   f"/uploads/{os.path.basename(self.frame_path)}" if self.frame_path else None,
        }

with app.app_context():
    db.create_all()

# ── YOLO loader ───────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
    return _model

# ── Config ────────────────────────────────────────────────────────────────────
# ROOT FIX: Lowered slightly to capture side-on bottles which have lower confidence
CONF_THRESHOLD  = 0.30 
PAD             = 15
TARGET_CLASS    = "bottle"

# ── Enhanced bbox refinement ────────────────────────────────────────────────
def refine_bbox_with_edges(frame_bgr, x1, y1, x2, y2, img_h, img_w):
    # ROOT FIX: Dynamic Search Padding based on Aspect Ratio
    box_w, box_h = x2 - x1, y2 - y1
    # If width > height, expand X more (for sideways bottles)
    mult_x = 0.6 if box_w > box_h else 0.4
    mult_y = 0.6 if box_h > box_w else 0.4
    
    sx1 = max(x1 - int(box_w * mult_x), 0)
    sy1 = max(y1 - int(box_h * mult_y), 0)
    sx2 = min(x2 + int(box_w * mult_x), img_w)
    sy2 = min(y2 + int(box_h * mult_y), img_h)

    search_region = frame_bgr[sy1:sy2, sx1:sx2]
    if search_region.size == 0: return x1, y1, x2, y2

    # ROOT FIX: Black-Hat Filter to highlight dark objects on any background
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    enhanced = cv2.add(gray, blackhat) # Boosts dark bottle features
    
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(blurred, threshold1=max(otsu_thresh * 0.3, 10), threshold2=otsu_thresh)
    
    # Use larger kernel to bridge structural gaps in the bottle body
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return max(x1-PAD,0), max(y1-PAD,0), min(x2+PAD,img_w), min(y2+PAD,img_h)

    cx_min, cy_min, cx_max, cy_max = img_w, img_h, 0, 0
    min_area = (box_w * box_h) * 0.1
    
    found_big = False
    for c in contours:
        if cv2.contourArea(c) > min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            cx_min, cy_min = min(cx_min, sx1+bx), min(cy_min, sy1+by)
            cx_max, cy_max = max(cx_max, sx1+bx+bw), max(cy_max, sy1+by+bh)
            found_big = True

    if not found_big: return x1, y1, x2, y2

    return max(min(x1, cx_min)-PAD, 0), max(min(y1, cy_min)-PAD, 0), \
           min(max(x2, cx_max)+PAD, img_w), min(max(y2, cy_max)+PAD, img_h)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(frame_bgr, source="webcam"):
    model = get_model()
    # ROOT FIX: augment=True enables TTA (Test-Time Augmentation) 
    # This flips and scales the image internally to catch odd angles/rotations
    results = model(frame_bgr, conf=CONF_THRESHOLD, augment=True, verbose=False)[0]

    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    detections = []

    bottle_boxes = [b for b in results.boxes if model.names[int(b.cls[0])] == TARGET_CLASS]

    frame_path = None
    if bottle_boxes:
        frame_fname = f"{uuid.uuid4().hex}.jpg"
        frame_path = os.path.join(UPLOAD_DIR, frame_fname)
        cv2.imwrite(frame_path, annotated)

    for box in bottle_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf, label = float(box.conf[0]), TARGET_CLASS

        rx1, ry1, rx2, ry2 = refine_bbox_with_edges(frame_bgr, x1, y1, x2, y2, h, w)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(annotated, f"BOTTLE {conf:.2f}", (rx1, max(ry1-10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        crop_fname = f"{uuid.uuid4().hex}_crop.jpg"
        crop_path = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, crop)

        with app.app_context():
            det = Detection(source=source, class_name=label, confidence=conf,
                            bbox_x1=rx1, bbox_y1=ry1, bbox_x2=rx2, bbox_y2=ry2,
                            crop_path=crop_path, frame_path=frame_path)
            db.session.add(det)
            db.session.commit()
            detections.append(det.to_dict())

    return annotated, detections

# ── Camera & Routes (Standard Flask Logic) ───────────────────────────────────
_camera = None
def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
    return _camera

def generate_frames():
    while True:
        cam = get_camera()
        ok, frame = cam.read()
        if not ok: break
        annotated, _ = run_inference(frame)
        _, buf = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/detections")
def api_detections():
    q = Detection.query.order_by(Detection.timestamp.desc()).limit(50)
    return jsonify([r.to_dict() for r in q.all()])

@app.route("/crops/<path:fname>")
def serve_crop(fname): return send_from_directory(CROP_DIR, fname)

if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)