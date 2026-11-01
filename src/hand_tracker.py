"""
Hand Tracker Module.

Provides robust 21-landmark hand detection using Google MediaPipe.
Designed to be compatible with both modern MediaPipe Tasks Vision API (>=0.10.0)
and legacy MediaPipe Solutions API (<0.10.0).
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Standard 21 Hand Connections for Visual Skeleton Rendering
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index Finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky Finger & Palm Base
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class NormalizedLandmark:
    """Represents a single normalized hand landmark coordinate."""
    x: float
    y: float
    z: float = 0.0


class LandmarkSmoother:
    """
    Exponential Moving Average (EMA) filter for 21 hand landmarks.
    Eliminates sub-pixel landmark jitter on distant hands (e.g. 2m away).
    """

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = float(alpha)
        self.prev_landmarks: Optional[List[NormalizedLandmark]] = None

    def smooth(self, landmarks: List[NormalizedLandmark]) -> List[NormalizedLandmark]:
        if not landmarks:
            self.reset()
            return []

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(landmarks):
            self.prev_landmarks = [
                NormalizedLandmark(x=lm.x, y=lm.y, z=getattr(lm, "z", 0.0))
                for lm in landmarks
            ]
            return landmarks

        smoothed: List[NormalizedLandmark] = []
        for prev, curr in zip(self.prev_landmarks, landmarks):
            curr_z = getattr(curr, "z", 0.0)
            sx = self.alpha * curr.x + (1.0 - self.alpha) * prev.x
            sy = self.alpha * curr.y + (1.0 - self.alpha) * prev.y
            sz = self.alpha * curr_z + (1.0 - self.alpha) * prev.z
            smoothed.append(NormalizedLandmark(x=sx, y=sy, z=sz))

        self.prev_landmarks = smoothed
        return smoothed

    def reset(self) -> None:
        self.prev_landmarks = None


class HandTracker:
    """
    Encapsulates MediaPipe hand detection and landmark tracking.
    Extracts 21 3D landmarks per hand and provides rendering utilities.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.40,
        model_path: Optional[str] = None,
        smooth_landmarks: bool = True,
        smoothing_alpha: float = 0.65,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self.smoother = LandmarkSmoother(alpha=smoothing_alpha) if smooth_landmarks else None

        # Determine path to hand_landmarker.task model
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = str(base_dir / "assets" / "hand_landmarker.task")
        self.model_path = model_path

        # Detection storage
        self.detected_hands_landmarks: List[List[NormalizedLandmark]] = []
        self.detected_hands_handedness: List[str] = []

        # Backend selector: 'tasks' (modern MediaPipe) or 'solutions' (legacy MediaPipe)
        self.backend: str = "none"
        self._detector: Any = None

        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize MediaPipe HandLandmarker or fallback to mp.solutions.hands."""
        # Attempt 1: Modern MediaPipe Tasks Vision API (MediaPipe 0.10.x+)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            if os.path.exists(self.model_path):
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._detector = HandLandmarker.create_from_options(options)
                self.backend = "tasks"
                return
        except (ImportError, AttributeError) as e:
            pass

        # Attempt 2: Legacy MediaPipe Solutions API (MediaPipe < 0.10.0)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.backend = "solutions"
                return
        except Exception:
            pass

        raise RuntimeError(
            f"Failed to initialize MediaPipe Hand Landmarker. "
            f"Tasks backend requires model file at: {self.model_path}. "
            f"Please verify model file exists and dependencies are installed."
        )

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> np.ndarray:
        """
        Process a video frame, detect hand landmarks, and optionally render them.

        Args:
            frame: BGR input image from OpenCV (np.ndarray).
            draw: If True, draws skeleton connections and joint landmarks on frame.

        Returns:
            np.ndarray: The original or annotated frame.
        """
        self.detected_hands_landmarks.clear()
        self.detected_hands_handedness.clear()

        if frame is None or frame.size == 0:
            return frame

        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "tasks":
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self._detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for hand_idx, raw_landmarks in enumerate(detection_result.hand_landmarks):
                    landmarks = [
                        NormalizedLandmark(x=lm.x, y=lm.y, z=getattr(lm, "z", 0.0))
                        for lm in raw_landmarks
                    ]
                    self.detected_hands_landmarks.append(landmarks)

                    # Extract handedness if available
                    handedness_str = "Right"
                    if (
                        detection_result.handedness
                        and hand_idx < len(detection_result.handedness)
                    ):
                        handedness_categories = detection_result.handedness[hand_idx]
                        if handedness_categories:
                            handedness_str = handedness_categories[0].category_name
                    self.detected_hands_handedness.append(handedness_str)

        elif self.backend == "solutions":
            results = self._detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    landmarks = [
                        NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        for lm in hand_lms.landmark
                    ]
                    self.detected_hands_landmarks.append(landmarks)

                    handedness_str = "Right"
                    if (
                        results.multi_handedness
                        and hand_idx < len(results.multi_handedness)
                    ):
                        handedness_str = results.multi_handedness[hand_idx].classification[0].label
                    self.detected_hands_handedness.append(handedness_str)

        # Apply temporal smoothing to primary hand if enabled
        if self.detected_hands_landmarks:
            if self.smoother is not None:
                self.detected_hands_landmarks[0] = self.smoother.smooth(
                    self.detected_hands_landmarks[0]
                )
        else:
            if self.smoother is not None:
                self.smoother.reset()

        # Draw landmarks and connections on frame if requested
        if draw and self.detected_hands_landmarks:
            self.draw_landmarks(frame)

        return frame

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """
        Draw landmarks and connecting bones onto the OpenCV frame.

        Args:
            frame: Target BGR image array.
        """
        h, w, _ = frame.shape

        for landmarks in self.detected_hands_landmarks:
            pixel_pts: List[Tuple[int, int]] = []
            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_pts.append((px, py))

            # 1. Draw bone connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(pixel_pts) and end_idx < len(pixel_pts):
                    pt1 = pixel_pts[start_idx]
                    pt2 = pixel_pts[end_idx]
                    cv2.line(frame, pt1, pt2, (40, 220, 40), 2, cv2.LINE_AA)

            # 2. Draw keypoint joints
            for i, (px, py) in enumerate(pixel_pts):
                # Highlight fingertips in bright cyan/yellow
                if i in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 4, (0, 165, 255), -1, cv2.LINE_AA)

    def get_hand_count(self) -> int:
        """Return the number of hands detected in the last processed frame."""
        return len(self.detected_hands_landmarks)

    def get_landmarks(self, hand_index: int = 0) -> Optional[List[NormalizedLandmark]]:
        """
        Retrieve the 21 normalized landmarks for the requested hand.

        Args:
            hand_index: Index of detected hand (0 for primary hand).

        Returns:
            List of 21 NormalizedLandmark objects, or None if no hand detected.
        """
        if 0 <= hand_index < len(self.detected_hands_landmarks):
            return self.detected_hands_landmarks[hand_index]
        return None

    def get_handedness(self, hand_index: int = 0) -> str:
        """Return 'Right' or 'Left' for the requested hand (default: 'Right')."""
        if 0 <= hand_index < len(self.detected_hands_handedness):
            return self.detected_hands_handedness[hand_index]
        return "Right"

    def close(self) -> None:
        """Release any internal detector resources."""
        if self._detector is not None and hasattr(self._detector, "close"):
            try:
                self._detector.close()
            except Exception:
                pass

