"""
Mobile Device Detector
─────────────────────
Strategy
  1. If a YOLOv4-tiny ONNX/weights model is present in models/ → use it (accurate).
  2. Fallback: lightweight rectangle-ratio heuristic using contour analysis.
     This catches phones held in frame as bright rectangular objects.

Place yolov4-tiny.cfg + yolov4-tiny.weights (or yolov4-tiny.onnx) inside  models/
to activate YOLO detection.  Download from:
  https://github.com/AlexeyAB/darknet (tiny weights)
"""

import os
import cv2
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_COCO_MOBILE_IDS = {67}   # COCO class 67 = "cell phone"

# Confidence / NMS thresholds
_CONF_THRESH = 0.40
_NMS_THRESH  = 0.45


class MobileDetector:
    def __init__(self):
        self._net   = None
        self._mode  = "heuristic"
        self._names = []
        self._output_layers: list[str] = []
        self._try_load_yolo()

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame) -> dict:
        """
        Returns:
            {
                "detected": bool,
                "boxes":    list of (x, y, w, h) in pixel coords,
                "mode":     "yolo" | "heuristic"
            }
        """
        if self._mode == "yolo":
            return self._yolo_detect(frame)
        return self._heuristic_detect(frame)

    # ── YOLO loader ───────────────────────────────────────────────────────────
    def _try_load_yolo(self):
        cfg_path     = os.path.join(_MODELS_DIR, "yolov4-tiny.cfg")
        weights_path = os.path.join(_MODELS_DIR, "yolov4-tiny.weights")
        names_path   = os.path.join(_MODELS_DIR, "coco.names")

        if not (os.path.exists(cfg_path) and os.path.exists(weights_path)):
            return   # stay in heuristic mode

        try:
            self._net = cv2.dnn.readNet(weights_path, cfg_path)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names        = self._net.getLayerNames()
            unconnected        = self._net.getUnconnectedOutLayers()
            self._output_layers = [layer_names[i - 1] for i in unconnected.flatten()]

            if os.path.exists(names_path):
                with open(names_path) as f:
                    self._names = [ln.strip() for ln in f.readlines()]

            self._mode = "yolo"
            print("[MobileDetector] YOLOv4-tiny loaded successfully.")
        except Exception as e:
            print(f"[MobileDetector] YOLO load failed ({e}), using heuristic fallback.")

    # ── YOLO inference ────────────────────────────────────────────────────────
    def _yolo_detect(self, frame) -> dict:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outs = self._net.forward(self._output_layers)

        boxes, confidences, class_ids = [], [], []

        for out in outs:
            for detection in out:
                scores    = detection[5:]
                class_id  = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if class_id not in _COCO_MOBILE_IDS:
                    continue
                if confidence < _CONF_THRESH:
                    continue
                cx, cy = int(detection[0] * w), int(detection[1] * h)
                bw, bh = int(detection[2] * w), int(detection[3] * h)
                boxes.append([cx - bw // 2, cy - bh // 2, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, _CONF_THRESH, _NMS_THRESH)

        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(tuple(boxes[i]))

        return {"detected": len(final_boxes) > 0, "boxes": final_boxes, "mode": "yolo"}

    # ── Heuristic fallback ────────────────────────────────────────────────────
    def _heuristic_detect(self, frame) -> dict:
        """
        Detect bright rectangular objects (likely a phone screen) using
        edge → contour analysis.  Fast but produces occasional false positives.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        frame_area = h * w
        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: phone-sized objects (0.5 % – 25 % of frame)
            if not (0.005 * frame_area < area < 0.25 * frame_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh else 0

            # Phone aspect ratios: ~0.45–0.7 (portrait) or ~1.4–2.2 (landscape)
            is_portrait  = 0.40 <= aspect <= 0.72
            is_landscape = 1.38 <= aspect <= 2.30

            if not (is_portrait or is_landscape):
                continue

            # Brightness inside rect — screens are typically bright
            roi         = gray[y: y + bh, x: x + bw]
            mean_bright = float(np.mean(roi))
            if mean_bright < 80:
                continue

            boxes.append((x, y, bw, bh))

        return {"detected": len(boxes) > 0, "boxes": boxes, "mode": "heuristic"}
