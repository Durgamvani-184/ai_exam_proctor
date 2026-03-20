# 🛡️ AI Exam Proctoring System

A professional, real-time exam cheating detection system built with Python, OpenCV, MediaPipe, and Flask.

---

## Features

| Feature | Details |
|---|---|
| **Face Detection** | Detects 0, 1, or multiple faces using MediaPipe FaceMesh |
| **Gaze / Head Tracking** | Detects looking left, right, up, down, and face off-centre |
| **Mobile Detection** | YOLOv4-tiny (if model files present) or heuristic contour fallback |
| **Alert Manager** | Rate-limited, severity-aware alerts (normal / warning / critical) |
| **Event Logger** | Rotating daily log files written to `logs/` |
| **Live Dashboard** | Flask + SSE — real-time video feed, event log, status gauge, stats |

---

## Project Structure

```
ai_exam_proctor/
├── app.py                      # Flask app & frame pipeline
├── requirements.txt
├── README.md
│
├── detectors/
│   ├── __init__.py
│   ├── face_detector.py        # MediaPipe FaceMesh — face count + landmarks
│   ├── gaze_tracker.py         # Head pose / gaze direction analysis
│   └── mobile_detector.py      # YOLO or heuristic phone detection
│
├── utils/
│   ├── __init__.py
│   ├── alert_manager.py        # Rate-limited alert system
│   └── logger.py               # Rotating file logger
│
├── models/                     # Place YOLOv4-tiny weights here (optional)
│   └── (yolov4-tiny.cfg + yolov4-tiny.weights + coco.names)
│
├── logs/                       # Auto-created; daily rotating log files
│
├── templates/
│   └── index.html              # Dashboard HTML
│
└── static/
    ├── css/style.css           # Dashboard styles
    └── js/dashboard.js         # SSE client + UI logic
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
```

### 3. Open browser

```
http://127.0.0.1:5000
```

---

## Optional: Enable YOLO Mobile Detection

The mobile detector falls back to a fast heuristic (contour analysis) if no model files are found.
To enable accurate YOLOv4-tiny detection, download the following files and place them in `models/`:

| File | Source |
|---|---|
| `yolov4-tiny.cfg` | https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg |
| `yolov4-tiny.weights` | https://github.com/AlexeyAB/darknet/releases |
| `coco.names` | https://github.com/AlexeyAB/darknet/blob/master/data/coco.names |

The system will automatically detect and load the model on next startup.

---

## Detection Logic

### Violation Priority (highest → lowest)

1. **📱 Mobile Device** — phone/tablet in frame
2. **⚠️ No Face** — student not visible
3. **👥 Multiple Faces** — more than one person
4. **👁️ Gaze Violation** — looking left / right / down / off-centre

### Alert Severities

| Severity | Colour | Cooldown |
|---|---|---|
| Normal | Green | — |
| Warning | Amber | 4 s |
| Critical | Red | 2 s |

---

## API Endpoints

| Route | Description |
|---|---|
| `GET /` | Dashboard UI |
| `GET /video` | MJPEG video stream |
| `GET /logs` | Server-Sent Events log stream |
| `GET /state` | Current state as JSON |
| `GET /alerts` | Alert history as JSON |

---

## Requirements

- Python 3.9+
- Webcam
- OpenCV, MediaPipe, Flask, NumPy (see `requirements.txt`)

---

## Author

Upgraded Exam Proctoring System — based on the original project by Durgam Vani.
