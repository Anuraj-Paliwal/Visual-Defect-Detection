# SENTINEL AI - Defect Detection System

<div align="center">

![SENTINEL AI](https://img.shields.io/badge/SENTINEL-AI-00FFAA?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)

**Real-time multi-camera defect detection system for quality control inspection**

[Features](#features) • [Demo](#demo) • [Installation](#installation) • [Usage](#usage) • [API](#api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

SENTINEL AI is a real-time defect detection system designed for industrial quality control. It uses a **two-stage AI pipeline** to detect and classify defects in manufactured products (specifically aluminum cans) using multiple camera angles.

### Two-Stage Detection Pipeline

1. **Stage 1 - Object Detection**: YOLOv8 model detects objects (cans) in the camera feed and crops them
2. **Stage 2 - Defect Classification**: Specialized models classify each crop as "perfect" or "defective"

### Key Capabilities

- 🎥 Multi-camera support (top view + side view)
- 🔄 Real-time processing with configurable frame skipping
- 📊 Live dashboard with detection history
- 🎛️ Configurable capture throttling to prevent duplicate detections
- 📱 Mobile camera support via IP webcam apps
- 💾 Automatic image storage and JSON logging
- ✅ Manual review workflow (accept/reject detections)

---

## ✨ Features

### Real-Time Detection
- **Multi-angle inspection**: Supports both top-view and side-view cameras
- **Live video streaming**: MJPEG streams with annotated bounding boxes
- **Automatic capture**: Throttled image capture (max 3 per object, 5-second cooldown)
- **Frame skipping**: Process every 3rd frame for performance optimization

### Intelligent Classification
- **Two-stage pipeline**: Separate models for detection and classification
- **Angle-specific models**: Different classifiers for top and side views
- **Background processing**: Asynchronous crop classification thread
- **Confidence scoring**: Shows both detection and classification confidence

### User Interface
- **Live Feed Page**: Monitor multiple cameras simultaneously
- **Dashboard**: Review detections with full metadata
- **Accept/Reject Workflow**: Manual quality control review
- **Batch Operations**: Reject all pending detections at once

### Data Management
- **Persistent storage**: Crops and full frames saved to disk
- **JSON logging**: Structured detection data with timestamps
- **Image retention**: Keep last 100 detections automatically
- **File cleanup**: Automatic deletion on rejection

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SENTINEL AI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────┐      ┌───────────┐                               │
│  │  TOP CAM  │      │ SIDE CAM  │                               │
│  │ (Phone 1) │      │ (Phone 2) │                               │
│  └─────┬─────┘      └─────┬─────┘                               │
│        │                   │                                     │
│        └───────┬───────────┘                                     │
│                │                                                 │
│         ┌──────▼──────┐                                          │
│         │   FLASK     │                                          │
│         │   SERVER    │                                          │
│         └──────┬──────┘                                          │
│                │                                                 │
│    ┌───────────┴───────────┐                                    │
│    │                       │                                    │
│ ┌──▼────────┐      ┌───────▼──────┐                            │
│ │  Stage 1  │      │   Stage 2    │                            │
│ │ Detection │─────▶│Classification│                            │
│ │  (YOLO)   │      │    (YOLO)    │                            │
│ └───────────┘      └──────┬───────┘                            │
│                           │                                     │
│              ┌────────────┴────────────┐                        │
│              │                         │                        │
│         ┌────▼─────┐            ┌─────▼──────┐                 │
│         │  Crops   │            │   JSON     │                 │
│         │ Storage  │            │  Database  │                 │
│         └──────────┘            └────────────┘                 │
│                                                                 │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │  Live Feed   │          │  Dashboard   │                    │
│  │     Page     │          │     Page     │                    │
│  └──────────────┘          └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Camera Feed → Frame Skip Filter → YOLO Detection → Crop & Save → 
Background Classifier → JSON Update → Dashboard Display → User Review
```

---

## 📦 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, Windows, or macOS
- **RAM**: 8GB minimum (16GB recommended for multiple cameras)
- **GPU**: CUDA-compatible GPU recommended (optional but faster)

### Software Dependencies

```bash
# Core frameworks
opencv-python>=4.5.0
ultralytics>=8.0.0
flask>=2.0.0

# AI/ML libraries
torch>=1.10.0
numpy>=1.21.0

# Model hub
huggingface-hub>=0.10.0
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentinel-ai.git
cd sentinel-ai
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import cv2; import torch; from ultralytics import YOLO; print('✓ All dependencies installed')"
```

---

## 📱 Hardware Setup

### Option 1: USB Webcams

1. Connect USB cameras to your computer
2. Note the device indices (usually 0, 1, 2, etc.)
3. Update `active_cameras` in `app.py`:

```python
active_cameras = [
    {
        "index": 0,  # First USB camera
        "name": "TOP CAM",
        "type": "top"
    },
    {
        "index": 1,  # Second USB camera
        "name": "SIDE CAM",
        "type": "side"
    },
]
```

### Option 2: Mobile Phone Cameras (IP Webcam)

**Recommended for flexible positioning**

#### Android Setup:

1. **Install IP Webcam App**
   - Download from [Google Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
   - Free and widely compatible

2. **Configure the App**
   - Open IP Webcam
   - Scroll to bottom and tap "Start Server"
   - Note the IP address (e.g., `http://192.168.1.9:8080`)

3. **Update Configuration**

```python
active_cameras = [
    {
        "index": "http://192.168.1.9:8080/video",  # Phone 1
        "name": "TOP CAM (Phone 1)",
        "type": "top"
    },
    {
        "index": "http://192.168.1.10:8080/video",  # Phone 2
        "name": "SIDE CAM (Phone 2)",
        "type": "side"
    },
]
```

#### iOS Setup:

1. **Install IP Camera Lite**
   - Download from App Store
   - Similar functionality to Android version

2. **Enable Server and Note URL**

**Network Requirements:**
- All devices must be on the same WiFi network
- Disable any VPNs
- Check firewall settings if connection fails

---

## ⚙️ Configuration

### Key Parameters

Edit these in `app.py`:

```python
# Detection Settings
CONF_THRESHOLD = 0.30          # Minimum confidence for detections (0.0-1.0)
PAD = 15                       # Padding around crops (pixels)

# Performance Settings
FRAME_SKIP = 3                 # Process every Nth frame (higher = faster)

# Capture Control
MAX_CAPTURES_PER_OBJECT = 3    # Max captures per detected object
COOLDOWN_SECONDS = 5           # Seconds between capture sessions
```

### Model Paths

Models are automatically downloaded from Hugging Face:

```python
# Stage 1: Can detection
DETECT_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="candetect/train/weights/last.pt"
)

# Stage 2: Top-view classifier
TOP_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="topview/train2/weights/last.pt"
)

# Stage 2: Side-view classifier
SIDE_MODEL_PATH = hf_hub_download(
    repo_id="Azu-nyan/Defect_detect_can",
    filename="sideview/train/weights/last.pt"
)
```

### Directory Structure

```
sentinel-ai/
├── app.py                 # Main Flask application
├── templates/
│   ├── live.html         # Live camera feed page
│   └── dashboard.html    # Detection review dashboard
├── static/
│   └── css/
│       └── style.css     # UI styling
├── uploads/              # Full-frame images (auto-created)
├── crops/                # Cropped detections (auto-created)
├── detections.json       # Detection log (auto-created)
└── requirements.txt      # Python dependencies
```

---

## 🎮 Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`

### Accessing the Interface

#### Live Feed Page
Navigate to: `http://localhost:5000/`

**Features:**
- Add/remove cameras dynamically
- Start/stop individual camera feeds
- View live annotated video streams
- Adjust capture cooldown settings
- Click on running cameras for fullscreen view

#### Dashboard Page
Navigate to: `http://localhost:5000/dashboard`

**Features:**
- View all detections with thumbnails
- See detailed metadata (confidence scores, timestamps, camera info)
- Review classification results (Stage 1 + Stage 2)
- Accept or reject individual detections
- Batch reject all pending detections
- View full-frame and crop images

### Workflow

1. **Setup Cameras**
   - Go to Live Feed page
   - Click "Add Camera" (+ card)
   - Scan for available cameras
   - Add top and side cameras

2. **Start Detection**
   - Click "START" button on each camera
   - Watch live feed with bounding boxes
   - Detections are automatically captured and saved

3. **Review Results**
   - Switch to Dashboard page
   - Wait for classification to complete (shows "PROCESSING" initially)
   - Review detection details
   - Accept good detections or reject false positives

4. **Monitor Statistics**
   - View real-time stats: Total, Defective, Perfect, Pending
   - Track system performance

---

## 📡 API Reference

### Camera Management

#### Scan for Cameras
```http
GET /api/cameras/scan
```

**Response:**
```json
{
  "cameras": [
    {"index": 0, "name": "Camera 0", "resolution": "1920x1080"},
    {"index": 1, "name": "Camera 1", "resolution": "1280x720"}
  ]
}
```

#### Get Active Cameras
```http
GET /api/cameras/active
```

#### Add Camera
```http
POST /api/cameras/add
Content-Type: application/json

{
  "index": 0,
  "name": "TOP CAM"
}
```

#### Remove Camera
```http
POST /api/cameras/remove
Content-Type: application/json

{
  "index": 0
}
```

### Camera Control

#### Start Camera
```http
POST /api/camera/start_one
Content-Type: application/json

{
  "index": 0
}
```

#### Stop Camera
```http
POST /api/camera/stop_one
Content-Type: application/json

{
  "index": 0
}
```

#### Get Camera Status
```http
GET /api/camera/status?cam=0
```

### Detections

#### Get All Detections
```http
GET /api/detections
```

**Response:**
```json
[
  {
    "id": "uuid-string",
    "timestamp": "2024-04-20 14:30:45",
    "camera_id": 0,
    "angle": "top",
    "class_name": "can",
    "confidence": 0.95,
    "status": "pending",
    "result": "defective",
    "cls_label": "defect",
    "cls_confidence": 0.87,
    "processed": true,
    "crop_url": "/crops/abc123_crop.jpg",
    "frame_url": "/uploads/abc123.jpg"
  }
]
```

#### Accept/Reject Detection
```http
POST /api/decision
Content-Type: application/json

{
  "id": "uuid-string",
  "decision": "accept"  # or "reject"
}
```

#### Reject All Pending
```http
POST /api/decision/reject_all
```

### Settings

#### Set Cooldown
```http
POST /api/set_cooldown
Content-Type: application/json

{
  "cooldown": 5.0
}
```

### Media Access

#### View Crop
```http
GET /crops/<filename>
```

#### View Full Frame
```http
GET /uploads/<filename>
```

#### Video Stream
```http
GET /video_feed?cam=<camera_index>
```

---

## 📁 Project Structure

```
sentinel-ai/
│
├── app.py                          # Main Flask application
│   ├── Camera management
│   ├── YOLO model loading
│   ├── Two-stage detection pipeline
│   ├── Background processing thread
│   └── API routes
│
├── templates/
│   ├── live.html                   # Live camera feed interface
│   │   ├── Dynamic camera grid
│   │   ├── Camera controls
│   │   └── Fullscreen modal
│   │
│   └── dashboard.html              # Detection review interface
│       ├── Statistics cards
│       ├── Detail panel
│       └── Detection list
│
├── static/
│   └── css/
│       └── style.css               # Custom UI styling
│
├── uploads/                        # Full-frame captures (auto-created)
├── crops/                          # Cropped detections (auto-created)
├── detections.json                 # Detection log (auto-created)
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected

**Problem:** Camera doesn't appear in scan results

**Solutions:**
- Check USB connection (for webcams)
- Verify camera permissions in OS settings
- Try different USB ports
- For IP cameras: ensure same WiFi network
- Check firewall settings

```bash
# Test camera manually
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

#### Model Download Fails

**Problem:** Hugging Face download timeout or error

**Solutions:**
- Check internet connection
- Verify Hugging Face is accessible
- Manually download models:

```bash
# Install huggingface CLI
pip install huggingface_hub[cli]

# Login (if needed)
huggingface-cli login

# Download models
huggingface-cli download Azu-nyan/Defect_detect_can
```

#### High CPU/Memory Usage

**Problem:** System running slow

**Solutions:**
- Increase `FRAME_SKIP` value (process fewer frames)
- Reduce camera resolution in IP webcam app
- Use GPU acceleration:

```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### IP Camera Connection Issues

**Problem:** Can't connect to phone camera

**Checklist:**
- [ ] Phone and computer on same WiFi
- [ ] IP Webcam server running on phone
- [ ] Correct IP address in config
- [ ] Port 8080 not blocked by firewall
- [ ] No VPN active on either device

**Test connection:**
```bash
# Test if server is reachable
curl http://192.168.1.9:8080
```

#### No Detections Appearing

**Problem:** Camera running but nothing detected

**Solutions:**
- Lower `CONF_THRESHOLD` in app.py
- Check camera angle (should clearly see objects)
- Verify proper lighting conditions
- Ensure objects are in frame
- Check model compatibility with your objects

---

## 🧪 Testing Mode

For testing classification without cameras:

1. **Uncomment test injection code** in `app.py`:

```python
def inject_test_detection():
    test_crop = "t1.jpeg"   # Put test image in /crops folder
    test_frame = "test.jpg"

    det = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": 999,
        "angle": "top",  # or "side"
        "class_name": "can",
        "confidence": 1.0,
        "status": "pending",
        "result": None,
        "cls_label": None,
        "cls_confidence": None,
        "processed": False,
        "crop_url": f"/crops/{test_crop}",
        "frame_url": f"/uploads/{test_frame}",
    }
    save_detection_json(det)
    print("✅ Test detection injected")

# In main:
if __name__ == "__main__":
    get_models()
    inject_test_detection()  # 👈 UNCOMMENT THIS
    threading.Thread(target=process_crops, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
```

2. **Place test image** in `crops/` folder as `t1.jpeg`
3. **Run server** and check Dashboard for classification result

---

## 🎯 Performance Optimization

### Speed vs Accuracy Trade-offs

```python
# Fast (lower accuracy)
CONF_THRESHOLD = 0.20
FRAME_SKIP = 5

# Balanced (recommended)
CONF_THRESHOLD = 0.30
FRAME_SKIP = 3

# Accurate (slower)
CONF_THRESHOLD = 0.40
FRAME_SKIP = 1
```

### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU (if GPU causing issues)
export CUDA_VISIBLE_DEVICES=-1
```

### Memory Management

```python
# Limit detection history (in app.py)
detections = detections[:50]  # Keep only 50 latest
```

---

## 📊 Model Information

### Detection Model (Stage 1)
- **Purpose**: Locate cans in frame
- **Architecture**: YOLOv8
- **Input**: Full camera frame
- **Output**: Bounding boxes with confidence scores

### Classification Models (Stage 2)
- **Top-View Classifier**: Specialized for overhead inspection
- **Side-View Classifier**: Specialized for lateral inspection
- **Architecture**: YOLOv8-based classification
- **Input**: Cropped can images
- **Output**: "perfect" / "defective" / "unknown" with confidence

---
### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black app.py
```