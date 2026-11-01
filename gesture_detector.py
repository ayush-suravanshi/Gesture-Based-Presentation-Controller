"""
Gesture Detector Module.

Integrates HandTracker with GestureRecognizer to form a cohesive vision pipeline.
Accepts raw video frames and outputs annotated frames along with gesture metadata.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from gesture_recognizer import GestureRecognizer
from src.hand_tracker import HandTracker, NormalizedLandmark


class GestureDetector:
    """
    Cohesive hand detection and gesture interpretation pipeline.
    Combines computer vision hand tracking with geometric gesture analysis.
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        model_path: Optional[str] = None,
        smooth_landmarks: bool = config.ENABLE_LANDMARK_SMOOTHING,
        smoothing_alpha: float = config.LANDMARK_SMOOTHING_ALPHA,
        history_frames: int = config.GESTURE_HISTORY_FRAMES,
    ) -> None:
        self.tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            smooth_landmarks=smooth_landmarks,
            smoothing_alpha=smoothing_alpha,
        )
        self.recognizer = GestureRecognizer(history_length=history_frames)

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single video frame.

        Args:
            frame: Input BGR image from OpenCV.
            draw: Whether to render landmark skeletons onto the frame.

        Returns:
            Tuple of:
                1. Annotated BGR frame.
                2. Recognition metadata dictionary for primary hand.
        """
        # Step 1: Detect hands and draw skeleton
        annotated_frame = self.tracker.find_hands(frame, draw=draw)

        # Step 2: Extract landmarks and handedness for the primary hand
        primary_landmarks = self.tracker.get_landmarks(hand_index=0)
        handedness = self.tracker.get_handedness(hand_index=0)

        # Step 3: Classify the gesture
        gesture_info = self.recognizer.recognize(primary_landmarks, handedness=handedness)
        gesture_info["hand_count"] = self.tracker.get_hand_count()
        gesture_info["landmarks"] = primary_landmarks

        return annotated_frame, gesture_info

    def close(self) -> None:
        """Release underlying detector resources."""
        self.tracker.close()

