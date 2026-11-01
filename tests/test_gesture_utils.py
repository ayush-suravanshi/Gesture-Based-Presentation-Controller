"""
Unit tests for Gesture Utilities and Finger-State Calculations.

Tests geometric functions and gesture classification using synthetic 21-landmark hand models.
Does not require any webcam or external hardware.
"""

from typing import Dict, List, NamedTuple
import pytest

from src.gesture_utils import (
    calculate_distance,
    is_thumb_up,
    is_index_finger_up,
    is_middle_finger_up,
    is_ring_finger_up,
    is_pinky_up,
    is_open_palm,
    is_closed_fist,
    get_finger_states,
    recognize_gesture,
    GESTURE_NEXT_SLIDE,
    GESTURE_PREV_SLIDE,
    GESTURE_RESUME,
    GESTURE_PAUSE,
    GESTURE_UNKNOWN,
)


class Point3D(NamedTuple):
    x: float
    y: float
    z: float = 0.0


def create_synthetic_hand(
    thumb_up: bool = False,
    index_up: bool = False,
    middle_up: bool = False,
    ring_up: bool = False,
    pinky_up: bool = False,
    scale: float = 1.0,
    handedness: str = "Right",
) -> List[Point3D]:
    """
    Generate synthetic 21 hand landmarks matching MediaPipe's normalized coordinate space.
    In image space: y=0.0 is top (up), y=1.0 is bottom (down).
    """
    wrist = Point3D(0.5, 0.8, 0.0)

    # Helper to generate MCP, PIP, DIP, TIP along an x-column
    def make_finger(x_base: float, is_up: bool):
        mcp = Point3D(x_base, 0.60, 0.0)
        if is_up:
            # Extended upward towards top of image (smaller y)
            pip = Point3D(x_base, 0.45, 0.0)
            dip = Point3D(x_base, 0.35, 0.0)
            tip = Point3D(x_base, 0.22, 0.0)
        else:
            # Curled downward into palm (larger y, close to wrist)
            pip = Point3D(x_base, 0.55, 0.0)
            dip = Point3D(x_base, 0.63, 0.0)
            tip = Point3D(x_base, 0.66, 0.0)
        return mcp, pip, dip, tip

    # Index (MCP 5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
    idx_mcp, idx_pip, idx_dip, idx_tip = make_finger(0.42, index_up)
    mid_mcp, mid_pip, mid_dip, mid_tip = make_finger(0.50, middle_up)
    rng_mcp, rng_pip, rng_dip, rng_tip = make_finger(0.58, ring_up)
    pnk_mcp, pnk_pip, pnk_dip, pnk_tip = make_finger(0.66, pinky_up)

    # Thumb (1..4)
    thumb_cmc = Point3D(0.44, 0.74, 0.0)
    thumb_mcp = Point3D(0.38, 0.68, 0.0)
    if thumb_up:
        # Extended outward away from pinky
        thumb_ip = Point3D(0.30, 0.62, 0.0)
        thumb_tip = Point3D(0.20, 0.55, 0.0)
    else:
        # Folded inward against palm towards pinky
        thumb_ip = Point3D(0.42, 0.66, 0.0)
        thumb_tip = Point3D(0.46, 0.65, 0.0)

    landmarks = [
        wrist,        # 0
        thumb_cmc,    # 1
        thumb_mcp,    # 2
        thumb_ip,     # 3
        thumb_tip,    # 4
        idx_mcp,      # 5
        idx_pip,      # 6
        idx_dip,      # 7
        idx_tip,      # 8
        mid_mcp,      # 9
        mid_pip,      # 10
        mid_dip,      # 11
        mid_tip,      # 12
        rng_mcp,      # 13
        rng_pip,      # 14
        rng_dip,      # 15
        rng_tip,      # 16
        pnk_mcp,      # 17
        pnk_pip,      # 18
        pnk_dip,      # 19
        pnk_tip,      # 20
    ]

    # Apply scaling relative to wrist if scale != 1.0
    if scale != 1.0:
        scaled = []
        for pt in landmarks:
            sx = wrist.x + (pt.x - wrist.x) * scale
            sy = wrist.y + (pt.y - wrist.y) * scale
            scaled.append(Point3D(sx, sy, pt.z))
        return scaled

    return landmarks


# ==============================================================================
# TESTS FOR GEOMETRIC DISTANCE & EXTRACTION
# ==============================================================================
def test_calculate_distance():
    """Verify 2D and 3D Euclidean distance calculation."""
    p1 = (0.0, 0.0, 0.0)
    p2 = (3.0, 4.0, 0.0)
    assert calculate_distance(p1, p2, include_z=False) == pytest.approx(5.0)

    p3 = Point3D(0.0, 0.0, 0.0)
    p4 = Point3D(1.0, 2.0, 2.0)
    assert calculate_distance(p3, p4, include_z=True) == pytest.approx(3.0)


# ==============================================================================
# TESTS FOR INDIVIDUAL FINGER EXTENSION DETECTION
# ==============================================================================
def test_individual_finger_detection():
    """Test individual finger detection across each digit."""
    # Only index extended
    hand = create_synthetic_hand(index_up=True)
    assert is_index_finger_up(hand) is True
    assert is_middle_finger_up(hand) is False
    assert is_ring_finger_up(hand) is False
    assert is_pinky_up(hand) is False

    # Only middle extended
    hand_mid = create_synthetic_hand(middle_up=True)
    assert is_index_finger_up(hand_mid) is False
    assert is_middle_finger_up(hand_mid) is True
    assert is_ring_finger_up(hand_mid) is False
    assert is_pinky_up(hand_mid) is False

    # Only ring extended
    hand_rng = create_synthetic_hand(ring_up=True)
    assert is_ring_finger_up(hand_rng) is True
    assert is_index_finger_up(hand_rng) is False

    # Only pinky extended
    hand_pnk = create_synthetic_hand(pinky_up=True)
    assert is_pinky_up(hand_pnk) is True
    assert is_index_finger_up(hand_pnk) is False


def test_finger_states_dictionary():
    """Test get_finger_states returns expected Boolean dictionary."""
    hand = create_synthetic_hand(thumb_up=True, index_up=True, middle_up=False)
    states = get_finger_states(hand)
    assert states["thumb"] is True
    assert states["index"] is True
    assert states["middle"] is False
    assert states["ring"] is False
    assert states["pinky"] is False


# ==============================================================================
# TESTS FOR TARGET GESTURES
# ==============================================================================
def test_gesture_next_slide():
    """GESTURE 1: Index finger up, other fingers down -> NEXT_SLIDE."""
    hand = create_synthetic_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
    gesture = recognize_gesture(hand)
    assert gesture == GESTURE_NEXT_SLIDE


def test_gesture_prev_slide():
    """GESTURE 2: Index and middle fingers up, others down -> PREV_SLIDE."""
    hand = create_synthetic_hand(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
    gesture = recognize_gesture(hand)
    assert gesture == GESTURE_PREV_SLIDE


def test_gesture_resume_open_palm():
    """GESTURE 3: Open palm (all fingers up) -> RESUME."""
    hand = create_synthetic_hand(
        thumb_up=True, index_up=True, middle_up=True, ring_up=True, pinky_up=True
    )
    assert is_open_palm(hand) is True
    assert is_closed_fist(hand) is False
    assert recognize_gesture(hand) == GESTURE_RESUME


def test_gesture_pause_closed_fist():
    """GESTURE 4: Closed fist (all fingers folded) -> PAUSE."""
    hand = create_synthetic_hand(
        thumb_up=False, index_up=False, middle_up=False, ring_up=False, pinky_up=False
    )
    assert is_closed_fist(hand) is True
    assert is_open_palm(hand) is False
    assert recognize_gesture(hand) == GESTURE_PAUSE


# ==============================================================================
# TESTS FOR UNKNOWN OR AMBIGUOUS GESTURES
# ==============================================================================
def test_unknown_gestures_handling():
    """Test that ambiguous hand shapes do not trigger accidental slide commands."""
    # Three fingers up (Index + Middle + Ring)
    three_fingers = create_synthetic_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=False)
    assert recognize_gesture(three_fingers) == GESTURE_UNKNOWN

    # Rock-on gesture (Index + Pinky up)
    rock_on = create_synthetic_hand(index_up=True, pinky_up=True)
    assert recognize_gesture(rock_on) == GESTURE_UNKNOWN

    # Only pinky up
    pinky_only = create_synthetic_hand(pinky_up=True)
    assert recognize_gesture(pinky_only) == GESTURE_UNKNOWN

    # None or insufficient landmarks
    assert recognize_gesture(None) == GESTURE_UNKNOWN
    assert recognize_gesture([]) == GESTURE_UNKNOWN
    assert recognize_gesture([Point3D(0, 0)] * 10) == GESTURE_UNKNOWN


# ==============================================================================
# TESTS FOR SCALE INVARIANCE (DISTANCE FROM CAMERA)
# ==============================================================================
def test_scale_invariance():
    """Verify detection remains accurate whether hand is close or far from camera."""
    # Far from camera (small hand, scale=0.6)
    far_hand = create_synthetic_hand(index_up=True, scale=0.6)
    assert recognize_gesture(far_hand) == GESTURE_NEXT_SLIDE

    # Close to camera (large hand, scale=1.4)
    close_hand = create_synthetic_hand(index_up=True, middle_up=True, scale=1.4)
    assert recognize_gesture(close_hand) == GESTURE_PREV_SLIDE

    # 2-meter standing distance (small hand, scale=0.30)
    distant_2m_hand = create_synthetic_hand(index_up=True, scale=0.30)
    assert recognize_gesture(distant_2m_hand) == GESTURE_NEXT_SLIDE

    # 2.5-meter distance (scale=0.25)
    distant_v_sign = create_synthetic_hand(index_up=True, middle_up=True, scale=0.25)
    assert recognize_gesture(distant_v_sign) == GESTURE_PREV_SLIDE


# ==============================================================================
# TESTS FOR DISTANCE ESTIMATION
# ==============================================================================
def test_distance_estimation():
    """Verify distance estimation calculates reasonable meters and range labels."""
    from src.gesture_utils import estimate_hand_distance

    # None or empty landmarks
    dist_none, label_none = estimate_hand_distance(None)
    assert dist_none == 0.0
    assert label_none == "No Hand"

    # Close hand (scale=1.5 -> distance < 1.0m)
    close_hand = create_synthetic_hand(scale=1.5)
    dist_close, label_close = estimate_hand_distance(close_hand)
    assert dist_close < 1.0
    assert "Close" in label_close

    # Normal presentation distance (~2.0m)
    # At normal scale (palm_span ~ 0.05), distance is ~2.0m
    hand_2m = create_synthetic_hand(scale=0.26)
    dist_2m, label_2m = estimate_hand_distance(hand_2m)
    assert 1.6 <= dist_2m <= 2.5
    assert "Optimal" in label_2m


def test_loosely_curled_finger_not_extended():
    """Verify that a loosely folded finger (straightness < 0.60) is classified as NOT extended."""
    # Hand with extended index finger and loosely curled middle finger
    hand = create_synthetic_hand(index_up=True, middle_up=False)
    assert is_index_finger_up(hand) is True
    assert is_middle_finger_up(hand) is False
    assert recognize_gesture(hand) == GESTURE_NEXT_SLIDE


