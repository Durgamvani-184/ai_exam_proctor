"""
Face Detector — uses MediaPipe FaceMesh to count faces and extract landmarks.
"""

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, max_faces: int = 4, min_detection_conf: float = 0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame) -> dict:
        """
        Returns:
            {
                "count":     int,          # number of faces found
                "landmarks": list | None,  # landmarks of the first face (or None)
                "all_landmarks": list,     # landmarks of all detected faces
            }
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"count": 0, "landmarks": None, "all_landmarks": []}

        all_lm = results.multi_face_landmarks
        return {
            "count":         len(all_lm),
            "landmarks":     all_lm[0].landmark,   # primary face
            "all_landmarks": [f.landmark for f in all_lm],
        }
