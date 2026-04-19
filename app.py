import os
import cv2
import uuid
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory, render_template
from ultralytics import YOLO

camera_running = False
app = Flask(__name__)
FRAME_SKIP = 3
frame_count = {}

# ── Capture Control ─────────────────────────────────────────────
MAX_CAPTURES_PER_OBJECT = 3
COOLDOWN_SECONDS = 5

object_tracker = {
    "count": 0,
    "last_capture_time": 0
}

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CROP_DIR   = os.path.join(BASE_DIR, "crops")
JSON_PATH  = os.path.join(BASE_DIR, "detections.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s.pt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    with open(JSON_PATH, "w") as f:
        json.dump([], f)

active_cameras = [{"index": 0, "name": "CAM 1"}]
camera_registry_lock = threading.Lock()

_cameras = {} 
_cam_lock = threading.Lock()

_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
    return _model

CONF_THRESHOLD = 0.30
PAD = 15
TARGET_CLASS = "bottle"
json_lock = threading.Lock()

def save_detection_json(data):
    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)
        detections.insert(0, data)
        detections = detections[:100]
        with open(JSON_PATH, "w") as f:
            json.dump(detections, f, indent=2)

def run_inference(frame_bgr, source="webcam", cam_id=0):
    # ── Guard: skip inference entirely if camera is off ──────────
    if not camera_running:
        return frame_bgr.copy(), []

    model = get_model()
    results = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

    annotated = frame_bgr.copy()
    detections = []

    bottle_boxes = [b for b in results.boxes if model.names[int(b.cls[0])] == TARGET_CLASS]

    # Nothing detected → return early, no files written
    if not bottle_boxes:
        return annotated, detections

    current_time = time.time()

    # Reset counter after cooldown expires
    if current_time - object_tracker["last_capture_time"] > COOLDOWN_SECONDS:
        object_tracker["count"] = 0

    # Cooldown cap reached → skip saving, no files written
    if object_tracker["count"] >= MAX_CAPTURES_PER_OBJECT:
        # Still draw boxes on the live feed for visibility
        for box in bottle_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange = suppressed
        return annotated, detections

    # ── Save the full frame once per detection event ─────────────
    frame_fname = f"{uuid.uuid4().hex}.jpg"
    frame_path  = os.path.join(UPLOAD_DIR, frame_fname)
    cv2.imwrite(frame_path, annotated)

    for box in bottle_boxes:
        if object_tracker["count"] >= MAX_CAPTURES_PER_OBJECT:
            break

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        crop = frame_bgr[y1:y2, x1:x2]
        crop_fname = f"{uuid.uuid4().hex}_crop.jpg"
        crop_path  = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, crop)

        object_tracker["count"] += 1
        object_tracker["last_capture_time"] = current_time

        det = {
            "id":         str(uuid.uuid4()),
            "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id":  cam_id,
            "class_name": TARGET_CLASS,
            "confidence": round(conf, 3),
            "bbox":       [x1, y1, x2, y2],
            "crop_url":   f"/crops/{crop_fname}",
            "frame_url":  f"/uploads/{frame_fname}",
            "status":     "pending"
        }

        save_detection_json(det)
        detections.append(det)

    return annotated, detections

def get_camera(device_index):
    """Return (or open) a cv2.VideoCapture for the given device index."""
    with _cam_lock:
        cap = _cameras.get(device_index)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(device_index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _cameras[device_index] = cap
        return cap

def release_camera(device_index):
    with _cam_lock:
        cap = _cameras.pop(device_index, None)
        if cap is not None:
            cap.release()

@app.route("/api/cameras/scan")
def scan_cameras():
    """
    Probe device indices 0-9 and return any that open successfully.
    Returns name, index, and resolution for each working camera.
    """
    found = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found.append({
                "index": idx,
                "name": f"Camera {idx}",
                "resolution": f"{w}x{h}"
            })
    return jsonify({"cameras": found})

@app.route("/api/cameras/active")
def get_active_cameras():
    """Return the list of cameras the user has added."""
    with camera_registry_lock:
        return jsonify({"cameras": list(active_cameras)})

@app.route("/api/cameras/add", methods=["POST"])
def add_camera():
    """Add a camera device to the active list."""
    data = request.json
    idx  = int(data.get("index", 0))
    name = data.get("name", f"Camera {idx}")

    with camera_registry_lock:
        # avoid duplicates
        if not any(c["index"] == idx for c in active_cameras):
            active_cameras.append({"index": idx, "name": name})

    return jsonify({"status": "added", "index": idx, "name": name})

@app.route("/api/cameras/remove", methods=["POST"])
def remove_camera():
    """Remove a camera device from the active list and release its capture."""
    data = request.json
    idx  = int(data.get("index", -1))

    with camera_registry_lock:
        for cam in list(active_cameras):
            if cam["index"] == idx:
                active_cameras.remove(cam)

    release_camera(idx)
    return jsonify({"status": "removed", "index": idx})

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    global camera_running
    camera_running = True
    return jsonify({"status": "started"})

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    global camera_running
    camera_running = False
    # Release all open captures
    with _cam_lock:
        for cap in _cameras.values():
            cap.release()
        _cameras.clear()
    return jsonify({"status": "stopped"})

def generate_frames(device_index=0):
    global frame_count
    while True:
        if not camera_running:
            time.sleep(0.1)
            continue

        cam = get_camera(device_index)
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.1)
            continue

        key = device_index
        frame_count[key] = frame_count.get(key, 0) + 1

        if frame_count[key] % FRAME_SKIP != 0:
            _, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            continue

        annotated, _ = run_inference(frame, cam_id=device_index)
        _, buf = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    device_index = int(request.args.get("cam", 0))
    return Response(
        generate_frames(device_index),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def live():
    return render_template("live.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/set_cooldown", methods=["POST"])
def set_cooldown():
    global COOLDOWN_SECONDS
    data = request.json
    COOLDOWN_SECONDS = float(data.get("cooldown", 5))
    return jsonify({"status": "ok", "cooldown": COOLDOWN_SECONDS})

@app.route("/api/detections")
def api_detections():
    with open(JSON_PATH, "r") as f:
        return jsonify(json.load(f))

@app.route("/crops/<path:fname>")
def serve_crop(fname):
    return send_from_directory(CROP_DIR, fname)

@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname)

def _delete_detection_files(detection_list):
    """Delete frame and crop files for a list of detection dicts."""
    deleted_frames = set()
    for d in detection_list:
        # ── Delete full frame (once per unique frame_url) ─────────
        frame_url = d.get("frame_url")
        if frame_url and frame_url not in deleted_frames:
            # frame_url is like  /uploads/abc123.jpg
            # strip the leading slash and join with BASE_DIR
            rel = frame_url.lstrip("/").replace("/", os.sep)
            frame_path = os.path.join(BASE_DIR, rel)
            try:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    print(f"Deleted frame: {frame_path}")
                else:
                    print(f"Frame not found (already deleted?): {frame_path}")
            except Exception as e:
                print(f"Frame delete error: {e}")
            deleted_frames.add(frame_url)

        # ── Delete crop ───────────────────────────────────────────
        crop_url = d.get("crop_url")
        if crop_url:
            rel = crop_url.lstrip("/").replace("/", os.sep)
            crop_path = os.path.join(BASE_DIR, rel)
            try:
                if os.path.exists(crop_path):
                    os.remove(crop_path)
                    print(f"Deleted crop: {crop_path}")
                else:
                    print(f"Crop not found (already deleted?): {crop_path}")
            except Exception as e:
                print(f"Crop delete error: {e}")

@app.route("/api/decision", methods=["POST"])
def handle_decision():
    data     = request.json
    det_id   = data.get("id")
    decision = data.get("decision")

    if not det_id or decision not in ["accept", "reject"]:
        return jsonify({"error": "Invalid request"}), 400

    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)

        target = next((d for d in detections if d["id"] == det_id), None)
        if not target:
            return jsonify({"error": "Not found"}), 404

        frame_url = target.get("frame_url")

        if decision == "reject":
            # remove all detections that share the same source frame
            removed = [d for d in detections if d.get("frame_url") == frame_url]
            updated = [d for d in detections if d.get("frame_url") != frame_url]
        else:
            for d in detections:
                if d["id"] == det_id:
                    d["status"] = "accepted"
            updated = detections
            removed = []

        with open(JSON_PATH, "w") as f:
            json.dump(updated, f, indent=2)

    if decision == "reject":
        _delete_detection_files(removed)

    return jsonify({"status": "success"})

@app.route("/api/decision/reject_all", methods=["POST"])
def reject_all():
    """Reject and delete every detection still in 'pending' status."""
    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)

        pending = [d for d in detections if d.get("status") == "pending"]
        updated = [d for d in detections if d.get("status") != "pending"]

        with open(JSON_PATH, "w") as f:
            json.dump(updated, f, indent=2)

    _delete_detection_files(pending)
    return jsonify({"status": "success", "removed": len(pending)})

if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)