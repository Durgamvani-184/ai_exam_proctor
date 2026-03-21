"""
Gaze Tracker — analyses head pose / eye direction using FaceMesh landmarks.
Detects: looking left/right, looking down, face off-center.
"""


# MediaPipe canonical landmark indices
_LEFT_EYE  = 33
_RIGHT_EYE = 263
_NOSE_TIP  = 1
_CHIN      = 152
_FOREHEAD  = 10


class GazeTracker:
    def __init__(self,
                 horizontal_ratio_threshold: float = 2.0,
                 vertical_diff_threshold: float = 0.12,
                 center_margin: float = 0.30):
        self.h_ratio_thresh = horizontal_ratio_threshold
        self.v_diff_thresh  = vertical_diff_threshold
        self.center_margin  = center_margin

    def track(self, frame, landmarks) -> dict:
        """
        Args:
            frame:     BGR frame (used for shape only)
            landmarks: list of MediaPipe landmark objects for the primary face,
                       or None if no face was detected.

        Returns:
            {
                "violation":        bool,
                "message":          str,
                "direction_vector": (dx, dy) | None  — normalised direction for HUD arrow
            }
        """
        if landmarks is None:
            return {"violation": False, "message": "No Face", "direction_vector": None}

        lm = landmarks
        left  = lm[_LEFT_EYE]
        right = lm[_RIGHT_EYE]
        nose  = lm[_NOSE_TIP]

        # ── Horizontal head-turn check ─────────────────────────────────────
        left_dist  = abs(nose.x - left.x)
        right_dist = abs(nose.x - right.x)

        if left_dist > 0 and right_dist > 0:
            ratio = left_dist / right_dist
            if ratio > self.h_ratio_thresh:
                return {
                    "violation": True,
                    "message": "Looking Left",
                    "direction_vector": (-1.0, 0.0),
                }
            if ratio < (1.0 / self.h_ratio_thresh):
                return {
                    "violation": True,
                    "message": "Looking Right",
                    "direction_vector": (1.0, 0.0),
                }

        # ── Vertical check (looking down at notes / phone) ─────────────────
        eye_y_avg    = (left.y + right.y) / 2.0
        vertical_diff = nose.y - eye_y_avg   # positive = nose below eyes (normal)

        if vertical_diff > self.v_diff_thresh + 0.06:
            return {
                "violation": True,
                "message": "Looking Down",
                "direction_vector": (0.0, 1.0),
            }

        if vertical_diff < -(self.v_diff_thresh):
            return {
                "violation": True,
                "message": "Looking Up",
                "direction_vector": (0.0, -1.0),
            }

        # ── Face off-centre check ──────────────────────────────────────────
        if nose.x < self.center_margin:
            return {
                "violation": True,
                "message": "Face Left of Frame",
                "direction_vector": (-1.0, 0.0),
            }
        if nose.x > (1.0 - self.center_margin):
            return {
                "violation": True,
                "message": "Face Right of Frame",
                "direction_vector": (1.0, 0.0),
            }

        return {"violation": False, "message": "Normal", "direction_vector": (0.0, 0.0)}
