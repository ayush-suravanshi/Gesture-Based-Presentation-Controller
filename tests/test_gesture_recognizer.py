"""
Unit tests for GestureRecognizer and high-level classification wrapper.
"""

import pytest
from gesture_recognizer import GestureRecognizer, GESTURE_DISPLAY_NAMES
from tests.test_gesture_utils import create_synthetic_hand, Point3D


def test_recognizer_no_hand():
    """Verify recognizer returns safe metadata when no hand is present."""
    recognizer = GestureRecognizer()
    res = recognizer.recognize(None)

    assert res["gesture"] == "UNKNOWN"
    assert res["hand_present"] is False
    assert res["command"] is None
    assert "No Hand" in res["display_name"]


def test_recognizer_next_slide():
    """Verify recognizer maps index finger up to NEXT_SLIDE command."""
    recognizer = GestureRecognizer()
    hand = create_synthetic_hand(index_up=True)
    res = recognizer.recognize(hand)

    assert res["gesture"] == "NEXT_SLIDE"
    assert res["command"] == "NEXT_SLIDE"
    assert res["hand_present"] is True


def test_recognizer_prev_slide():
    """Verify recognizer maps two fingers up to PREV_SLIDE command."""
    recognizer = GestureRecognizer()
    hand = create_synthetic_hand(index_up=True, middle_up=True)
    res = recognizer.recognize(hand)

    assert res["gesture"] == "PREV_SLIDE"
    assert res["command"] == "PREV_SLIDE"
    assert res["hand_present"] is True


def test_recognizer_resume_and_pause():
    """Verify open palm and closed fist commands."""
    recognizer = GestureRecognizer()

    # Open Palm
    palm = create_synthetic_hand(
        thumb_up=True, index_up=True, middle_up=True, ring_up=True, pinky_up=True
    )
    res_palm = recognizer.recognize(palm)
    assert res_palm["gesture"] == "RESUME"
    assert res_palm["command"] == "RESUME"

    # Closed Fist
    fist = create_synthetic_hand(
        thumb_up=False, index_up=False, middle_up=False, ring_up=False, pinky_up=False
    )
    res_fist = recognizer.recognize(fist)
    assert res_fist["gesture"] == "PAUSE"
    assert res_fist["command"] == "PAUSE"


def test_recognizer_majority_voting_for_distant_hands():
    """
    Test that at 2 meters, a 1-frame landmark flicker does NOT destroy the gesture recognition.
    With history_length=5, [NEXT, NEXT, UNKNOWN, NEXT, NEXT] remains NEXT_SLIDE!
    """
    recognizer = GestureRecognizer(history_length=5)
    hand_next = create_synthetic_hand(index_up=True)

    # Feed 2 frames of NEXT_SLIDE
    assert recognizer.recognize(hand_next)["gesture"] == "NEXT_SLIDE"
    assert recognizer.recognize(hand_next)["gesture"] == "NEXT_SLIDE"

    # Frame 3 has a momentary distant flutter (UNKNOWN)
    res_flicker = recognizer.recognize(None)
    # Because 2 out of 3 frames so far are NEXT_SLIDE, majority voting retains NEXT_SLIDE!
    assert res_flicker["gesture"] == "NEXT_SLIDE"

    # Frame 4 returns to NEXT_SLIDE
    assert recognizer.recognize(hand_next)["gesture"] == "NEXT_SLIDE"


def test_landmark_smoother():
    """Verify Exponential Moving Average (EMA) smoothing for landmark coordinates."""
    from src.hand_tracker import LandmarkSmoother, NormalizedLandmark

    smoother = LandmarkSmoother(alpha=0.6)

    # Frame 1: Initial position at x=0.5
    lms1 = [NormalizedLandmark(x=0.5, y=0.5, z=0.0)]
    smoothed1 = smoother.smooth(lms1)
    assert smoothed1[0].x == pytest.approx(0.5)

    # Frame 2: Jittered position at x=0.7
    # Expected: alpha * 0.7 + (1 - alpha) * 0.5 = 0.6 * 0.7 + 0.4 * 0.5 = 0.42 + 0.20 = 0.62
    lms2 = [NormalizedLandmark(x=0.7, y=0.5, z=0.0)]
    smoothed2 = smoother.smooth(lms2)
    assert smoothed2[0].x == pytest.approx(0.62)

    # Reset
    smoother.reset()
    assert smoother.prev_landmarks is None

