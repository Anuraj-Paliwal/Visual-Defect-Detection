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
from huggingface_hub import hf_hub_download

camera_states = {}   # { cam_index: True/False }
app = Flask(__name__)
FRAME_SKIP = 3
frame_count = {}

SIDE_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="sideview/train/weights/last.pt"
)
TOP_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="topview/train2/weights/last.pt"
)
DETECT_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="candetect/train/weights/last.pt"
)

# ── Capture Control ──────────────────────────────────────────────
MAX_CAPTURES_PER_OBJECT = 3
COOLDOWN_SECONDS        = 5

object_tracker = {}  # { cam_id: { count, last_capture_time } }

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CROP_DIR   = os.path.join(BASE_DIR, "crops")
JSON_PATH  = os.path.join(BASE_DIR, "detections.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

if not os.path.exists(JSON_PATH):
    with open(JSON_PATH, "w") as f:
        json.dump([], f)

# ── DEFAULT: both slots point to webcam (index 0) until you have  ─
#    two physical cameras. Change the "index" values then.          ─
active_cameras = [
    {
        "index": "http://192.168.1.9:8080/video",
        "name": "TOP CAM (Phone 1)",
        "type": "top"
    },
    {
        "index": "http://192.168.1.10:8080/video",
        "name": "SIDE CAM (Phone 2)",
        "type": "side"
    },
]
camera_registry_lock = threading.Lock()

_cameras    = {}
_cam_lock   = threading.Lock()
_models     = {}
_model_lock = threading.Lock()

class CropHeatmapGenerator:

    def __init__(self, img_bgr):
        self.img = img_bgr
        self.gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ─────────────────────────────
    # Option 5: Texture Variance
    # ─────────────────────────────
    def texture_heatmap(self):
        window = 15

        mean = cv2.blur(self.gray.astype(np.float32), (window, window))
        sqr_mean = cv2.blur(self.gray.astype(np.float32)**2, (window, window))
        var = np.maximum(sqr_mean - mean**2, 0)
        std = np.sqrt(var)

        std = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(std, cv2.COLORMAP_PLASMA)

        return heatmap

    # ─────────────────────────────
    # Option 6: Multi-scale
    # ─────────────────────────────
    def multisacle_heatmap(self):
        edges1 = cv2.Canny(cv2.GaussianBlur(self.gray, (3, 3), 0), 50, 150)
        edges2 = cv2.Canny(cv2.GaussianBlur(self.gray, (9, 9), 0), 30, 100)

        gx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        combined = (
            edges1 * 0.3 +
            edges2 * 0.2 +
            mag * 0.5
        )

        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_INFERNO)

        return heatmap
    
def save_heatmaps(crop_img, crop_fname_base):
    """
    Generates + saves heatmaps for a crop
    returns saved file paths
    """

    gen = CropHeatmapGenerator(crop_img)

    texture = gen.texture_heatmap()
    multi   = gen.multisacle_heatmap()

    texture_name = f"{crop_fname_base}_texture.jpg"
    multi_name   = f"{crop_fname_base}_multi.jpg"

    texture_path = os.path.join(CROP_DIR, texture_name)
    multi_path   = os.path.join(CROP_DIR, multi_name)

    cv2.imwrite(texture_path, texture)
    cv2.imwrite(multi_path, multi)

    return {
        "texture_heatmap": f"/crops/{texture_name}",
        "multi_heatmap": f"/crops/{multi_name}"
    }

def get_models():
    global _models
    if not _models:
        with _model_lock:
            if not _models:
                _models["detect"] = YOLO(DETECT_MODEL_PATH)
                _models["top"]    = YOLO(TOP_MODEL_PATH)
                _models["side"]   = YOLO(SIDE_MODEL_PATH)
    return _models


CONF_THRESHOLD = 0.30
PAD            = 15
json_lock      = threading.Lock()


def save_detection_json(data):
    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)
        detections.insert(0, data)
        detections = detections[:100]
        with open(JSON_PATH, "w") as f:
            json.dump(detections, f, indent=2)


def run_inference(frame_bgr, cam_id=0):
    """
    Stage 1 ONLY:
      - Detect cans in full frame
      - Save cropped images
      - Store metadata in JSON

    No defect classification happens here.
    """

    if not camera_states.get(cam_id, False):
        return frame_bgr.copy(), []

    models       = get_models()
    detect_model = models["detect"]

    # ── Stage 1: can detection ─────────────────────────────
    results    = detect_model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
    annotated  = frame_bgr.copy()
    detections = []

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, detections

    # ── Capture throttle ───────────────────────────────────
    tracker = object_tracker.setdefault(cam_id, {
        "count": 0,
        "last_capture_time": 0
    })

    current_time = time.time()

    if current_time - tracker["last_capture_time"] > COOLDOWN_SECONDS:
        tracker["count"] = 0

    # If limit reached → only draw boxes
    if tracker["count"] >= MAX_CAPTURES_PER_OBJECT:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        return annotated, detections

    # ── Resolve camera type ────────────────────────────────
    cam_type = "top"
    with camera_registry_lock:
        for cam in active_cameras:
            if cam["index"] == cam_id:
                cam_type = cam.get("type", "top")
                break

    # Save original frame once
    frame_fname = f"{uuid.uuid4().hex}.jpg"
    frame_path  = os.path.join(UPLOAD_DIR, frame_fname)
    cv2.imwrite(frame_path, frame_bgr)

    h, w = frame_bgr.shape[:2]

    # ── Process detections ─────────────────────────────────
    for box in boxes:
        if tracker["count"] >= MAX_CAPTURES_PER_OBJECT:
            break

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Draw bounding box ONLY (no classification)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            annotated, "CAN",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
        )

        # ── Crop with padding ─────────────────────────────
        x1p = max(0, x1 - PAD)
        y1p = max(0, y1 - PAD)
        x2p = min(w, x2 + PAD)
        y2p = min(h, y2 + PAD)

        crop = frame_bgr[y1p:y2p, x1p:x2p]

        # Save crop
# Save crop
        crop_id = uuid.uuid4().hex
        crop_fname = f"{crop_id}_crop.jpg"
        crop_path  = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, crop)

        # ─────────────────────────────
        # NEW: Generate heatmaps
        # ─────────────────────────────
        heatmap_paths = save_heatmaps(crop, crop_id)

        tracker["count"] += 1
        tracker["last_capture_time"] = current_time

        # ── JSON entry (NO classification yet) ─────────────
        det = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": cam_id,
            "angle": cam_type,
            "class_name": "can",
            "confidence": round(conf, 3),
            "status": "pending",

            # ❗ Stage 2 will fill these later
            "result": None,
            "cls_label": None,
            "cls_confidence": None,
            "processed": False,

            "crop_url": f"/crops/{crop_fname}",
            "heatmaps": {
                "texture": heatmap_paths["texture_heatmap"],
                "multi_scale": heatmap_paths["multi_heatmap"]
            },
            "frame_url": f"/uploads/{frame_fname}",
        }

        save_detection_json(det)
        detections.append(det)

    return annotated, detections

def process_crops():
    while True:
        with json_lock:
            with open(JSON_PATH, "r") as f:
                detections = json.load(f)

        updated = False

        for d in detections:
            # ✅ safer check (handles missing key too)
            if d.get("processed", False):
                continue

            crop_path = os.path.join(BASE_DIR, d["crop_url"].lstrip("/"))
            if not os.path.exists(crop_path):
                continue

            img = cv2.imread(crop_path)

            # ✅ select correct model
            model = get_models()["top"] if d["angle"] == "top" else get_models()["side"]

            result = model(img, verbose=False)[0]

            # ✅ DETECTION MODELS → use boxes
            if result.boxes is not None and len(result.boxes) > 0:

                # ✅ pick BEST box (highest confidence)
                boxes = result.boxes
                confs = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                best_box = boxes[best_idx]

                cid = int(best_box.cls[0])
                conf = float(best_box.conf[0])

                name = model.names.get(cid, "unknown").lower()

            else:
                name, conf = "unknown", 0.0

            # ✅ optional mapping (clean UI labels)
            if name in ["defect", "defective", "bad"]:
                final = "defective"
            elif name in ["perfect", "good", "ok", "positive"]:
                final = "perfect"
            else:
                final = "unknown"

            # ✅ update JSON
            d["cls_label"] = name
            d["cls_confidence"] = round(conf, 3)
            d["result"] = final
            d["processed"] = True

            updated = True

        if updated:
            with json_lock:
                with open(JSON_PATH, "w") as f:
                    json.dump(detections, f, indent=2)

        time.sleep(0.5)

# ── Camera helpers ────────────────────────────────────────────────
def get_camera(device_index):
    with _cam_lock:
        cap = _cameras.get(device_index)

        if cap is None or not cap.isOpened():

            cap = cv2.VideoCapture(device_index)

            if not cap.isOpened():
                print(f"[ERROR] Cannot open camera {device_index}")
                return None

            _cameras[device_index] = cap

        return cap

def release_camera(device_index):
    with _cam_lock:
        cap = _cameras.pop(device_index, None)
        if cap is not None:
            cap.release()


# ── API routes ────────────────────────────────────────────────────
@app.route("/api/cameras/scan")
def scan_cameras():
    found = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found.append({"index": idx, "name": f"Camera {idx}", "resolution": f"{w}x{h}"})
    return jsonify({"cameras": found})


@app.route("/api/cameras/active")
def get_active_cameras():
    with camera_registry_lock:
        return jsonify({"cameras": list(active_cameras)})


@app.route("/api/cameras/add", methods=["POST"])
def add_camera():
    data     = request.json
    idx      = int(data.get("index", 0))
    name     = data.get("name", f"Camera {idx}")
    cam_type = "top" if "TOP" in name.upper() else "side"

    with camera_registry_lock:
        # Allow same device index for top AND side (different angle models)
        if not any(c["index"] == idx and c["type"] == cam_type for c in active_cameras):
            active_cameras.append({"index": idx, "name": name, "type": cam_type})

    return jsonify({"status": "added", "index": idx, "name": name})


@app.route("/api/cameras/remove", methods=["POST"])
def remove_camera():
    data = request.json
    idx = data.get("index")

    with camera_registry_lock:
        for cam in list(active_cameras):
            if cam["index"] == idx:
                active_cameras.remove(cam)

    camera_states[idx] = False
    release_camera(idx)
    return jsonify({"status": "removed", "index": idx})


@app.route("/api/camera/start_one", methods=["POST"])
def start_one():
    data = request.json
    idx = data.get("index")

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return jsonify({"status": "error", "reason": f"Cannot open device {idx}"})
    cap.release()

    camera_states[idx] = True
    return jsonify({"status": "ok"})


@app.route("/api/camera/stop_one", methods=["POST"])
def stop_one():
    data = request.json
    idx = data.get("index")

    camera_states[idx] = False
    release_camera(idx)
    return jsonify({"status": "stopped"})


@app.route("/api/camera/status")
def camera_status():
    idx = request.args.get("cam")   # ✅ (no int)
    return jsonify({"index": idx, "running": camera_states.get(idx, False)})


def generate_frames(device_index=0):
    global frame_count
    while True:
        if not camera_states.get(device_index, False):
            time.sleep(0.1)
            continue

        cam = get_camera(device_index)
        ok, frame = cam.read()

        if not ok or frame is None:
            print(f"[WARN] Camera lost: {device_index}, reconnecting...")
            release_camera(device_index)
            time.sleep(1)
            continue

        frame_count[device_index] = frame_count.get(device_index, 0) + 1

        if frame_count[device_index] % FRAME_SKIP != 0:
            _, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            continue

        # Annotated frame has bounding boxes only; classification detail is JSON-only
        annotated, _ = run_inference(frame, cam_id=device_index)
        _, buf = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    device_index = request.args.get("cam")   # ✅
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
    for d in detection_list:

        # ── DELETE FRAME ─────────────────────────────
        frame_url = d.get("frame_url")
        if frame_url:
            try:
                frame_path = os.path.join(BASE_DIR, frame_url.lstrip("/"))
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            except Exception as e:
                print(f"Frame delete error: {e}")

        # ── DELETE CROP ──────────────────────────────
        crop_url = d.get("crop_url")
        if crop_url:
            try:
                crop_path = os.path.join(BASE_DIR, crop_url.lstrip("/"))
                if os.path.exists(crop_path):
                    os.remove(crop_path)
            except Exception as e:
                print(f"Crop delete error: {e}")

        # ── DELETE HEATMAPS (IMPORTANT FIX) ─────────
        heatmaps = d.get("heatmaps", {})
        for url in heatmaps.values():
            try:
                heatmap_path = os.path.join(BASE_DIR, url.lstrip("/"))
                if os.path.exists(heatmap_path):
                    os.remove(heatmap_path)
            except Exception as e:
                print(f"Heatmap delete error: {e}")

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
    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)

        pending = [d for d in detections if d.get("status") == "pending"]
        updated = [d for d in detections if d.get("status") != "pending"]

        with open(JSON_PATH, "w") as f:
            json.dump(updated, f, indent=2)

    _delete_detection_files(pending)
    return jsonify({"status": "success", "removed": len(pending)})

@app.route("/api/detection/heatmaps/<det_id>")
def get_heatmaps(det_id):

    with json_lock:
        with open(JSON_PATH, "r") as f:
            detections = json.load(f)

    det = next((d for d in detections if d["id"] == det_id), None)

    if not det:
        return jsonify({"error": "not found"}), 404

    return jsonify({
        "id": det_id,
        "heatmaps": det.get("heatmaps", {})
    })

# def inject_test_detection():
#     test_crop = "t1.jpeg"   # ← put your image in /crops folder
#     test_frame = "test.jpg"  # optional

#     det = {
#         "id": str(uuid.uuid4()),
#         "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
#         "camera_id": 999,
#         "angle": "top",  # 🔥 change to "top" or "side" to test both
#         "class_name": "can",
#         "confidence": 1.0,
#         "status": "pending",

#         "result": None,
#         "cls_label": None,
#         "cls_confidence": None,
#         "processed": False,

#         "crop_url": f"/crops/{test_crop}",
#         "frame_url": f"/uploads/{test_frame}",
#     }

#     save_detection_json(det)
#     print("✅ Test detection injected")

if __name__ == "__main__":
    get_models()
    # inject_test_detection()  # 👈 ADD THIS LINE
    threading.Thread(target=process_crops, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)