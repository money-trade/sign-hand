"""ตัวห่อ MediaPipe Holistic สำหรับตรวจจับร่างกายและมือ"""
import mediapipe as mp
import numpy as np

from .features import frame_from_results

_holistic = mp.solutions.holistic
_drawing = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles


class HolisticTracker:
    def __init__(self, static_image_mode=False, model_complexity=1):
        self._h = _holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb: np.ndarray):
        """รับภาพ RGB -> (results, frame) โดย frame = (points, mask, pose_vis) หรือ None"""
        rgb.flags.writeable = False
        results = self._h.process(rgb)
        h, w = rgb.shape[:2]
        return results, frame_from_results(results, (w, h))

    def close(self):
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_landmarks(bgr, results):
    """วาดโครงร่างกายและมือลงบนภาพ"""
    if results.pose_landmarks:
        _drawing.draw_landmarks(
            bgr, results.pose_landmarks, _holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_drawing.DrawingSpec(color=(180, 180, 180), thickness=2),
        )
    for hand, color in ((results.left_hand_landmarks, (255, 140, 0)),
                        (results.right_hand_landmarks, (0, 200, 255))):
        if hand:
            _drawing.draw_landmarks(
                bgr, hand, _holistic.HAND_CONNECTIONS,
                _drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                _drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )
