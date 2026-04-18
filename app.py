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
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

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
CONF_THRESHOLD  = 0.40
PAD             = 10
TARGET_CLASS    = "bottle"

# ── Edge-based bbox refinement ────────────────────────────────────────────────
def refine_bbox_with_edges(frame_bgr, x1, y1, x2, y2, img_h, img_w):
    """
    Expands a YOLO bounding box to fully contain the detected bottle by:
    1. Searching an enlarged search region around the YOLO box
    2. Running Canny edge detection in that region
    3. Finding the bounding rect of all significant contours
    4. Merging YOLO box + contour box and adding final padding

    Returns refined (rx1, ry1, rx2, ry2) clipped to image bounds.
    """
    # ── 1. Build a generously padded search region ────────────────────────────
    # Use 50% expansion of the YOLO box so we catch parts the model missed
    box_w = x2 - x1
    box_h = y2 - y1
    search_pad_x = max(int(box_w * 0.5), 30)
    search_pad_y = max(int(box_h * 0.5), 30)

    sx1 = max(x1 - search_pad_x, 0)
    sy1 = max(y1 - search_pad_y, 0)
    sx2 = min(x2 + search_pad_x, img_w)
    sy2 = min(y2 + search_pad_y, img_h)

    search_region = frame_bgr[sy1:sy2, sx1:sx2]
    if search_region.size == 0:
        return x1, y1, x2, y2

    # ── 2. Pre-process for edge detection ────────────────────────────────────
    gray    = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-threshold via Otsu for robust edge finding on any lighting
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred,
                      threshold1=max(otsu_thresh * 0.4, 10),
                      threshold2=otsu_thresh)

    # Morphological close to bridge small gaps in bottle edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ── 3. Find significant contours ─────────────────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No contours found – fall back to padded YOLO box
        return (max(x1 - PAD, 0), max(y1 - PAD, 0),
                min(x2 + PAD, img_w), min(y2 + PAD, img_h))

    # Keep only contours large enough to be part of a bottle (filter noise)
    min_contour_area = (box_w * box_h) * 0.01   # at least 1% of YOLO box area
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

    if not big_contours:
        big_contours = contours   # safety: use all if filter removes everything

    # ── 4. Merge all contour bounding boxes into one ──────────────────────────
    cx_min, cy_min = img_w, img_h
    cx_max, cy_max = 0, 0
    for c in big_contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        # coords are relative to search_region – convert to full-frame coords
        cx_min = min(cx_min, sx1 + bx)
        cy_min = min(cy_min, sy1 + by)
        cx_max = max(cx_max, sx1 + bx + bw)
        cy_max = max(cy_max, sy1 + by + bh)

    # ── 5. Union of YOLO box and contour box ─────────────────────────────────
    rx1 = min(x1, cx_min)
    ry1 = min(y1, cy_min)
    rx2 = max(x2, cx_max)
    ry2 = max(y2, cy_max)

    # ── 6. Final safety padding + clamp ──────────────────────────────────────
    rx1 = max(rx1 - PAD, 0)
    ry1 = max(ry1 - PAD, 0)
    rx2 = min(rx2 + PAD, img_w)
    ry2 = min(ry2 + PAD, img_h)

    return rx1, ry1, rx2, ry2


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(frame_bgr, source="webcam"):
    model    = get_model()
    results  = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

    h, w     = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    detections = []

    bottle_boxes = [
        box for box in results.boxes
        if model.names[int(box.cls[0])] == TARGET_CLASS
    ]

    frame_path = None
    if bottle_boxes:
        frame_fname = f"{uuid.uuid4().hex}.jpg"
        frame_path  = os.path.join(UPLOAD_DIR, frame_fname)
        cv2.imwrite(frame_path, annotated)

    for box in bottle_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        label = TARGET_CLASS

        # ── Refine crop region using edge detection ───────────────────────────
        rx1, ry1, rx2, ry2 = refine_bbox_with_edges(frame_bgr, x1, y1, x2, y2, h, w)

        # Draw the *refined* box on the annotated frame so you can verify it
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 200, 80), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}",
                    (rx1, max(ry1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)

        # Crop using the refined bbox
        crop = frame_bgr[ry1:ry2, rx1:rx2]

        crop_fname = f"{uuid.uuid4().hex}_crop.jpg"
        crop_path  = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, crop)

        with app.app_context():
            det = Detection(
                source=source,
                class_name=label,
                confidence=conf,
                bbox_x1=rx1, bbox_y1=ry1,
                bbox_x2=rx2, bbox_y2=ry2,
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
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    data  = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    _, detections = run_inference(frame, source="upload")
    return jsonify({"detections": detections})

@app.route("/api/detections")
def api_detections():
    q = Detection.query.order_by(Detection.timestamp.desc()).limit(100)
    return jsonify([r.to_dict() for r in q.all()])

@app.route("/api/stats")
def api_stats():
    from sqlalchemy import func
    rows = (db.session.query(Detection.class_name,
                             func.count(Detection.id).label("count"))
            .group_by(Detection.class_name).all())
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
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
