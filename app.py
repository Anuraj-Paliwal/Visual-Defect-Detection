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
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")   # auto-downloaded on first run

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
    source      = db.Column(db.String(64))       # "webcam" | "upload"
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
                _model = YOLO(MODEL_PATH)   # downloads yolov8n.pt automatically
    return _model

# ── Helpers ───────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.40
PAD = 10   # pixels of padding around each crop

def run_inference(frame_bgr, source="webcam"):
    """
    Run YOLOv8 on a BGR numpy frame.
    Returns (annotated_frame, list_of_detection_dicts).
    Saves crops and a full annotated frame to disk, persists to DB.
    """
    model = get_model()
    results = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    detections = []

    # Save full annotated frame once
    frame_fname = f"{uuid.uuid4().hex}.jpg"
    frame_path  = os.path.join(UPLOAD_DIR, frame_fname)
    cv2.imwrite(frame_path, annotated)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]

        # Draw bbox on annotated frame
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)

        # Crop with padding
        cx1 = max(x1 - PAD, 0)
        cy1 = max(y1 - PAD, 0)
        cx2 = min(x2 + PAD, w)
        cy2 = min(y2 + PAD, h)
        crop = frame_bgr[cy1:cy2, cx1:cx2]

        crop_fname = f"{uuid.uuid4().hex}_crop.jpg"
        crop_path  = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, crop)

        # Persist to DB
        with app.app_context():
            det = Detection(
                source=source,
                class_name=label,
                confidence=conf,
                bbox_x1=x1, bbox_y1=y1,
                bbox_x2=x2, bbox_y2=y2,
                crop_path=crop_path,
                frame_path=frame_path,
            )
            db.session.add(det)
            db.session.commit()
            detections.append(det.to_dict())

    return annotated, detections

# ── Webcam stream ─────────────────────────────────────────────────────────────
_camera      = None
_camera_lock = threading.Lock()

def get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        with _camera_lock:
            if _camera is None or not _camera.isOpened():
                _camera = cv2.VideoCapture(0)
    return _camera

def generate_frames():
    while True:
        cam = get_camera()
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        annotated, _ = run_inference(frame, source="webcam")

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload", methods=["POST"])
def upload():
    """Accept an image upload, run inference, return detections JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    data = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    _, detections = run_inference(frame, source="upload")
    return jsonify({"detections": detections})

@app.route("/api/detections")
def api_detections():
    """Return recent detections (last 100) with optional class filter."""
    cls_filter = request.args.get("class")
    q = Detection.query.order_by(Detection.timestamp.desc())
    if cls_filter:
        q = q.filter_by(class_name=cls_filter)
    rows = q.limit(100).all()
    return jsonify([r.to_dict() for r in rows])

@app.route("/api/stats")
def api_stats():
    """Aggregate counts per class for the dashboard."""
    from sqlalchemy import func
    rows = (db.session.query(Detection.class_name,
                             func.count(Detection.id).label("count"))
            .group_by(Detection.class_name)
            .all())
    return jsonify([{"class": r.class_name, "count": r.count} for r in rows])

@app.route("/crops/<path:fname>")
def serve_crop(fname):
    return send_from_directory(CROP_DIR, fname)

@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting server — loading YOLOv8 model …")
    get_model()   # warm up
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
