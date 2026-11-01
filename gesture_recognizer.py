"""
Gesture Recognizer Module.

Provides high-level gesture classification, human-readable descriptions,
and gesture confidence metrics.
"""

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple, Union

from src.gesture_utils import (
    GESTURE_NEXT_SLIDE,
    GESTURE_PAUSE,
    GESTURE_PREV_SLIDE,
    GESTURE_RESUME,
    GESTURE_UNKNOWN,
    estimate_hand_distance,
    get_finger_states,
    recognize_gesture,
)

# Human-readable labels and descriptions for display on HUD
GESTURE_DISPLAY_NAMES = {
    GESTURE_NEXT_SLIDE: "Index Pointing (Next Slide)",
    GESTURE_PREV_SLIDE: "Peace Sign (Previous Slide)",
    GESTURE_RESUME: "Open Palm (Resume Control)",
    GESTURE_PAUSE: "Closed Fist (Pause Control)",
    GESTURE_UNKNOWN: "Unknown / Neutral Hand Pose",
}

GESTURE_COMMAND_MAPPING = {
    GESTURE_NEXT_SLIDE: "NEXT_SLIDE",
    GESTURE_PREV_SLIDE: "PREV_SLIDE",
    GESTURE_RESUME: "RESUME",
    GESTURE_PAUSE: "PAUSE",
    GESTURE_UNKNOWN: None,
}


class GestureRecognizer:
    """
    Interprets raw hand landmarks and returns high-level gesture metadata,
    finger state breakdowns, distance estimation, and descriptive UI labels.
    Supports optional rolling-history majority voting to prevent frame drops at 2+ meters.
    """

    def __init__(self, history_length: int = 1) -> None:
        self.history_length = max(1, int(history_length))
        self.history: deque = deque(maxlen=self.history_length)

    def reset_history(self) -> None:
        """Clear temporal gesture history."""
        self.history.clear()

    def recognize(
        self,
        landmarks: Optional[Union[List[Any], Tuple[Any, ...]]],
        handedness: str = "Right",
    ) -> Dict[str, Any]:
        """
        Analyze hand landmarks and return a comprehensive classification result.

        Args:
            landmarks: 21 landmark items or None.
            handedness: 'Right' or 'Left'.

        Returns:
            dict with:
                - 'gesture': gesture identifier (e.g. 'NEXT_SLIDE')
                - 'raw_gesture': instantaneous gesture before temporal voting
                - 'display_name': readable label
                - 'command': presentation command or None
                - 'finger_states': dict of thumb, index, middle, ring, pinky booleans
                - 'hand_present': bool
                - 'handedness': str
                - 'distance_m': estimated distance in meters
                - 'distance_label': formatted distance indicator string
        """
        if landmarks is None or len(landmarks) < 21:
            self.history.append(GESTURE_UNKNOWN)
            # If no hand detected, stabilize as UNKNOWN unless prior frames had clear majority
            stabilized_gesture = GESTURE_UNKNOWN
            if self.history_length > 1 and len(self.history) >= 2:
                counts = Counter(self.history)
                most_common, freq = counts.most_common(1)[0]
                if freq >= (len(self.history) * 0.5) and most_common != GESTURE_UNKNOWN:
                    stabilized_gesture = most_common

            return {
                "gesture": stabilized_gesture,
                "raw_gesture": GESTURE_UNKNOWN,
                "display_name": "No Hand Detected",
                "command": GESTURE_COMMAND_MAPPING.get(stabilized_gesture),
                "finger_states": {
                    "thumb": False,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False,
                },
                "hand_present": False,
                "handedness": handedness,
                "distance_m": 0.0,
                "distance_label": "No Hand",
            }

        finger_states = get_finger_states(landmarks, handedness)
        raw_gesture = recognize_gesture(landmarks, handedness)
        self.history.append(raw_gesture)

        # Apply majority voting across recent frames if history_length > 1
        if self.history_length > 1:
            counts = Counter(self.history)
            most_common, freq = counts.most_common(1)[0]
            # Require at least 50% majority agreement
            if freq >= (len(self.history) * 0.5):
                final_gesture = most_common
            else:
                final_gesture = raw_gesture
        else:
            final_gesture = raw_gesture

        display_name = GESTURE_DISPLAY_NAMES.get(final_gesture, "Unknown Pose")
        command = GESTURE_COMMAND_MAPPING.get(final_gesture)
        dist_m, dist_label = estimate_hand_distance(landmarks)

        return {
            "gesture": final_gesture,
            "raw_gesture": raw_gesture,
            "display_name": display_name,
            "command": command,
            "finger_states": finger_states,
            "hand_present": True,
            "handedness": handedness,
            "distance_m": dist_m,
            "distance_label": dist_label,
        }

