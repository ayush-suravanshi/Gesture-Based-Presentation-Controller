"""
Gesture Utilities Module.

This module provides geometric and mathematical functions to analyze 21 MediaPipe
hand landmarks, calculate finger states (extended vs folded), and accurately classify
predefined presentation control gestures.

MediaPipe Hand Landmark Reference (21 Keypoints):
    0:  WRIST
    1:  THUMB_CMC,  2:  THUMB_MCP,  3:  THUMB_IP,   4:  THUMB_TIP
    5:  INDEX_MCP,  6:  INDEX_PIP,  7:  INDEX_DIP,  8:  INDEX_TIP
    9:  MIDDLE_MCP, 10: MIDDLE_PIP, 11: MIDDLE_DIP, 12: MIDDLE_TIP
    13: RING_MCP,   14: RING_PIP,   15: RING_DIP,   16: RING_TIP
    17: PINKY_MCP,  18: PINKY_PIP,  19: PINKY_DIP,  20: PINKY_TIP
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Landmark Index Constants for Readability
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Standard Gesture Identifier Strings
GESTURE_NEXT_SLIDE = "NEXT_SLIDE"
GESTURE_PREV_SLIDE = "PREV_SLIDE"
GESTURE_RESUME = "RESUME"
GESTURE_PAUSE = "PAUSE"
GESTURE_UNKNOWN = "UNKNOWN"


def _extract_coords(pt: Any) -> Tuple[float, float, float]:
    """
    Safely extract (x, y, z) coordinates from either an object with
    x, y, z attributes (MediaPipe Landmark) or a tuple/list of floats.
    """
    if hasattr(pt, "x") and hasattr(pt, "y"):
        z = getattr(pt, "z", 0.0)
        return float(pt.x), float(pt.y), float(z)
    elif isinstance(pt, (tuple, list)):
        x = float(pt[0])
        y = float(pt[1])
        z = float(pt[2]) if len(pt) > 2 else 0.0
        return x, y, z
    raise TypeError(f"Unsupported point representation: {type(pt)}")


def calculate_distance(pt1: Any, pt2: Any, include_z: bool = False) -> float:
    """
    Calculate the Euclidean distance between two 2D/3D points.

    Args:
        pt1: First point (landmark object or tuple/list).
        pt2: Second point (landmark object or tuple/list).
        include_z: If True, calculates 3D Euclidean distance; otherwise 2D (x, y).

    Returns:
        float: Euclidean distance between pt1 and pt2.
    """
    x1, y1, z1 = _extract_coords(pt1)
    x2, y2, z2 = _extract_coords(pt2)

    if include_z:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_finger_extended(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
) -> bool:
    """
    Determine if a standard finger (Index, Middle, Ring, or Pinky) is extended (UP).

    Uses a robust multi-criteria kinematic approach:
    1. Bone segment straightness: Ratio of direct MCP->TIP distance to total phalanx path length.
       Straight fingers have a ratio >= 0.70; curled fingers bend backwards with ratio < 0.60.
    2. Radial extension: Fingertip is farther from wrist than the PIP joint.
    3. Phalanx extension: Knuckle-to-tip distance exceeds knuckle-to-PIP distance by >1.35x.
    4. Curled guard: If tip is curled below DIP and PIP, or tucked towards wrist, finger is folded.
    5. Upright vertical orientation: If hand is upright (wrist below MCP), fingertip must be higher than PIP (tip_y < pip_y).

    Args:
        landmarks: List of 21 hand landmarks.
        mcp_idx: Metacarpophalangeal joint index (finger base).
        pip_idx: Proximal interphalangeal joint index.
        dip_idx: Distal interphalangeal joint index.
        tip_idx: Fingertip index.

    Returns:
        bool: True if the finger is extended; False if curled or folded.
    """
    if len(landmarks) < 21:
        return False

    wrist = landmarks[WRIST]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]

    _, wrist_y, _ = _extract_coords(wrist)
    _, mcp_y, _ = _extract_coords(mcp)
    _, pip_y, _ = _extract_coords(pip)
    _, dip_y, _ = _extract_coords(dip)
    _, tip_y, _ = _extract_coords(tip)

    # Calculate individual bone segment lengths (MCP -> PIP -> DIP -> TIP)
    d_mcp_pip = calculate_distance(mcp, pip)
    d_pip_dip = calculate_distance(pip, dip)
    d_dip_tip = calculate_distance(dip, tip)
    total_bone_length = d_mcp_pip + d_pip_dip + d_dip_tip

    if total_bone_length <= 0.001 or d_mcp_pip <= 0.001:
        return False

    d_mcp_tip = calculate_distance(mcp, tip)
    d_wrist_tip = calculate_distance(wrist, tip)
    d_wrist_pip = calculate_distance(wrist, pip)

    # Definite curled guard: if tip is curled below DIP and PIP (for upright hand)
    if tip_y >= dip_y and tip_y >= pip_y:
        return False

    # Definite curled guard: if tip is tucked closer to wrist than PIP
    if d_wrist_tip < (d_wrist_pip * 0.90):
        return False

    # 1. Straightness: ratio of direct knuckle-to-tip distance vs sum of individual phalanges
    straightness = d_mcp_tip / total_bone_length

    # 2. Radial extension: fingertip is further away from wrist than PIP joint
    radial_ext = d_wrist_tip > d_wrist_pip

    # 3. Phalanx extension: direct MCP-to-tip distance is significantly longer than MCP-to-PIP
    phalanx_ext = d_mcp_tip > (d_mcp_pip * 1.35)

    # 4. Upright check: if hand is pointing up (wrist below knuckle), tip must be higher than PIP
    wrist_is_bottom = wrist_y > mcp_y
    vertical_ok = (tip_y < pip_y) if wrist_is_bottom else True

    return (straightness >= 0.70) and radial_ext and phalanx_ext and vertical_ok


def is_thumb_up(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """
    Determine if the thumb is extended.

    Unlike other fingers, the thumb articulates sideways (horizontally).
    We evaluate extension using:
    1. Distance from thumb tip to pinky MCP base (landmark 17).
       When thumb is opened/extended, this span is significantly larger.
    2. Distance from thumb tip to wrist vs thumb IP to wrist.

    Args:
        landmarks: List of 21 hand landmarks.
        handedness: "Right" or "Left" (based on MediaPipe hand classification).

    Returns:
        bool: True if the thumb is extended; False otherwise.
    """
    if len(landmarks) < 21:
        return False

    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]
    thumb_mcp = landmarks[THUMB_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    wrist = landmarks[WRIST]

    dist_tip_pinky = calculate_distance(thumb_tip, pinky_mcp)
    dist_ip_pinky = calculate_distance(thumb_ip, pinky_mcp)

    dist_tip_wrist = calculate_distance(thumb_tip, wrist)
    dist_mcp_wrist = calculate_distance(thumb_mcp, wrist)

    # Thumb is extended away from palm
    return (dist_tip_pinky > dist_ip_pinky) and (dist_tip_wrist > dist_mcp_wrist * 1.1)


def is_index_finger_up(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """Check if the index finger is extended (UP)."""
    return is_finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)


def is_middle_finger_up(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """Check if the middle finger is extended (UP)."""
    return is_finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)


def is_ring_finger_up(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """Check if the ring finger is extended (UP)."""
    return is_finger_extended(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP)


def is_pinky_up(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """Check if the pinky finger is extended (UP)."""
    return is_finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)


def get_finger_states(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> Dict[str, bool]:
    """
    Get the Boolean state (True = UP/Extended, False = DOWN/Folded) of all 5 fingers.

    Returns:
        dict: {
            "thumb": bool,
            "index": bool,
            "middle": bool,
            "ring": bool,
            "pinky": bool
        }
    """
    if len(landmarks) < 21:
        return {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False,
        }

    return {
        "thumb": is_thumb_up(landmarks, handedness),
        "index": is_index_finger_up(landmarks, handedness),
        "middle": is_middle_finger_up(landmarks, handedness),
        "ring": is_ring_finger_up(landmarks, handedness),
        "pinky": is_pinky_up(landmarks, handedness),
    }


def is_open_palm(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """
    Check if the hand is an open palm.
    All 4 primary fingers (index, middle, ring, pinky) must be extended UP,
    and thumb is typically extended as well.
    """
    states = get_finger_states(landmarks, handedness)
    # Open palm requires at least all 4 main fingers up
    return states["index"] and states["middle"] and states["ring"] and states["pinky"]


def is_closed_fist(
    landmarks: Union[List[Any], Tuple[Any, ...]],
    handedness: str = "Right",
) -> bool:
    """
    Check if the hand is a closed fist.
    All 4 primary fingers (index, middle, ring, pinky) must be folded DOWN.
    """
    states = get_finger_states(landmarks, handedness)
    # Closed fist requires all 4 main fingers curled down
    return not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]


def recognize_gesture(
    landmarks: Optional[Union[List[Any], Tuple[Any, ...]]],
    handedness: str = "Right",
) -> str:
    """
    Recognize and classify the hand pose into one of the designated presentation gestures.

    Designated Gestures:
    1. NEXT_SLIDE: Index finger UP; middle, ring, pinky DOWN.
    2. PREV_SLIDE: Index and middle fingers UP; ring, pinky DOWN.
    3. RESUME: Open palm (all fingers extended).
    4. PAUSE: Closed fist (all fingers curled down).
    5. UNKNOWN: Hand detected, but does not match any of the designated control gestures.

    Args:
        landmarks: 21 landmarks of the detected hand, or None if no hand detected.
        handedness: Handedness label ("Right" or "Left").

    Returns:
        str: One of GESTURE_NEXT_SLIDE, GESTURE_PREV_SLIDE, GESTURE_RESUME,
             GESTURE_PAUSE, or GESTURE_UNKNOWN.
    """
    if landmarks is None or len(landmarks) < 21:
        return GESTURE_UNKNOWN

    states = get_finger_states(landmarks, handedness)

    idx_up = states["index"]
    mid_up = states["middle"]
    rng_up = states["ring"]
    pnk_up = states["pinky"]

    # 1. Open Palm -> Resume gesture control
    if idx_up and mid_up and rng_up and pnk_up:
        return GESTURE_RESUME

    # 2. Closed Fist -> Pause gesture control
    if not idx_up and not mid_up and not rng_up and not pnk_up:
        return GESTURE_PAUSE

    # 3. Index finger ONLY -> Move to next slide
    if idx_up and not mid_up and not rng_up and not pnk_up:
        return GESTURE_NEXT_SLIDE

    # 4. Index AND Middle fingers ONLY -> Move to previous slide
    if idx_up and mid_up and not rng_up and not pnk_up:
        return GESTURE_PREV_SLIDE

    # 5. Any other finger combination is safely considered UNKNOWN
    return GESTURE_UNKNOWN


def estimate_hand_distance(
    landmarks: Optional[Union[List[Any], Tuple[Any, ...]]],
) -> Tuple[float, str]:
    """
    Estimate physical distance (in meters) between presenter's hand and the webcam.
    Uses anthropometric palm scale (average adult palm height = ~10.5 cm)
    calibrated against typical webcam focal geometry.

    Args:
        landmarks: 21 hand landmarks.

    Returns:
        Tuple of (estimated_distance_meters, range_label_str)
        e.g. (2.05, "Optimal (2.0m)")
    """
    if landmarks is None or len(landmarks) < 21:
        return 0.0, "No Hand"

    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    palm_span = calculate_distance(wrist, middle_mcp)

    if palm_span <= 0.005:
        return 0.0, "Invalid"

    # Calibration: at ~2.0m distance, palm_span is ~0.050 - 0.055 of normalized height
    dist_m = round(0.105 / palm_span, 1)
    dist_m = max(0.4, min(4.5, dist_m))

    if dist_m < 1.0:
        label = f"{dist_m:.1f}m [Close]"
    elif dist_m <= 2.6:
        label = f"{dist_m:.1f}m [Optimal 2m]"
    else:
        label = f"{dist_m:.1f}m [Far]"

    return dist_m, label

