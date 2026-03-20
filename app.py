"""
AI Exam Proctoring System - Main Application
Author: Upgraded System
"""

import cv2
import time
import json
import numpy as np
from flask import Flask, Response, render_template, jsonify
from detectors.face_detector import FaceDetector
from detectors.gaze_tracker import GazeTracker
from detectors.mobile_detector import MobileDetector
from utils.alert_manager import AlertManager
from utils.logger import EventLogger

app = Flask(__name__)

# ─── Initialize Detectors ────────────────────────────────────────────────────
face_detector  = FaceDetector()
gaze_tracker   = GazeTracker()
mobile_detector = MobileDetector()
alert_manager  = AlertManager()
event_logger   = EventLogger()

# ─── Camera Init ─────────────────────────────────────────────────────────────
def init_camera():
    # On Windows, CAP_DSHOW often works better for some cameras
    backends = [cv2.CAP_DSHOW, None] 
    
    for backend in backends:
        for idx in range(3):
            if backend is not None:
                cam = cv2.VideoCapture(idx, backend)
            else:
                cam = cv2.VideoCapture(idx)
                
            if cam.isOpened():
                ret, frame = cam.read()
                if ret and frame is not None:
                    return cam
            cam.release()
            
    return cv2.VideoCapture(0)

camera = init_camera()

# ─── Shared State ─────────────────────────────────────────────────────────────
state = {
    "cheating": False,
    "message": "System Initialized",
    "severity": "normal",   # normal | warning | critical
    "face_count": 0,
    "mobile_detected": False,
}

# ─── Frame Generator ──────────────────────────────────────────────────────────
def generate_frames():
    global state, camera

    while True:
        if not camera.isOpened():
            camera = init_camera()

        success, frame = camera.read()

        # ── Handle bad / black frames ──────────────────────────────────────
        if not success or (frame is not None and np.mean(frame) < 5):
            state.update({
                "cheating": True,
                "message": "No Video Signal",
                "severity": "critical",
                "face_count": 0,
                "mobile_detected": False,
            })
            placeholder = _make_placeholder("NO VIDEO SIGNAL", "Check camera connection or privacy shutter.")
            yield _encode_frame(placeholder)
            time.sleep(1)
            camera = init_camera()
            continue

        # ── Run detectors ──────────────────────────────────────────────────
        face_result   = face_detector.detect(frame)
        gaze_result   = gaze_tracker.track(frame, face_result["landmarks"])
        mobile_result = mobile_detector.detect(frame)

        # ── Merge results into a single violation status ───────────────────
        violation, message, severity = _evaluate(face_result, gaze_result, mobile_result)

        state.update({
            "cheating": violation,
            "message": message,
            "severity": severity,
            "face_count": face_result["count"],
            "mobile_detected": mobile_result["detected"],
        })

        if violation:
            alert_manager.trigger(severity, message)
            event_logger.log(message, severity)

        # ── Annotate frame ─────────────────────────────────────────────────
        frame = _annotate(frame, state, gaze_result, mobile_result)

        yield _encode_frame(frame)


def _evaluate(face, gaze, mobile):
    """Combine detector outputs into one violation decision."""
    # Priority order: mobile > multiple faces > gaze
    if mobile["detected"]:
        return True, "📱 Mobile Device Detected", "critical"

    if face["count"] == 0:
        return True, "⚠️ No Face Detected", "critical"

    if face["count"] > 1:
        return True, f"👥 {face['count']} Faces Detected", "critical"

    if gaze["violation"]:
        return True, f"👁️ {gaze['message']}", "warning"

    return False, "✅ Normal", "normal"


def _annotate(frame, state, gaze, mobile):
    """Draw HUD overlay on frame."""
    h, w = frame.shape[:2]
    severity = state["severity"]

    # Header bar
    bar_color = {"normal": (0, 180, 0), "warning": (0, 165, 255), "critical": (0, 0, 220)}[severity]
    cv2.rectangle(frame, (0, 0), (w, 55), (10, 10, 10), -1)
    cv2.putText(frame, state["message"], (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, bar_color, 2)

    # Face count badge
    badge = f"Faces: {state['face_count']}"
    cv2.rectangle(frame, (w - 160, 8), (w - 10, 48), (30, 30, 30), -1)
    cv2.putText(frame, badge, (w - 150, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Mobile alert banner
    if mobile["detected"]:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "MOBILE DEVICE IN FRAME", (int(w / 2) - 160, h - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

    # Gaze direction arrow (if landmarks exist)
    if gaze.get("direction_vector"):
        cx, cy = int(w / 2), int(h / 2)
        dx, dy = gaze["direction_vector"]
        ex, ey = int(cx + dx * 60), int(cy + dy * 60)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

    return frame


def _make_placeholder(title, subtitle):
    """Create a black placeholder frame with text."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, title,    (60, 220), cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 50, 240), 2)
    cv2.putText(frame, subtitle, (60, 270), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (140, 140, 140), 1)
    return frame


def _encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ─── SSE Log Generator ────────────────────────────────────────────────────────
def generate_logs():
    last_msg = ""
    while True:
        if state["message"] != last_msg:
            last_msg = state["message"]
            payload = {
                "time": time.strftime("%H:%M:%S"),
                "status": state["message"],
                "severity": state["severity"],
                "cheating": state["cheating"],
                "face_count": state["face_count"],
                "mobile": state["mobile_detected"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        time.sleep(0.4)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    return Response(generate_logs(), mimetype="text/event-stream")

@app.route("/state")
def get_state():
    return jsonify(state)

@app.route("/alerts")
def get_alerts():
    return jsonify(alert_manager.get_history())

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
