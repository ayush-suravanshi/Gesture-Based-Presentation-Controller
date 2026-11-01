"""
Unit tests for CommandManager and PresentationController.

Tests gesture stabilization, hold timing, anti-repeat logic, cooldown enforcement,
state transitions (Active/Paused/Safe Mode), and mocks keyboard command dispatch.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.command_manager import CommandManager, SystemState
from src.gesture_utils import (
    GESTURE_NEXT_SLIDE,
    GESTURE_PAUSE,
    GESTURE_PREV_SLIDE,
    GESTURE_RESUME,
    GESTURE_UNKNOWN,
)
from presentation_controller import PresentationController


# ==============================================================================
# TESTS FOR GESTURE STABILIZATION & HOLD TIMING
# ==============================================================================
def test_gesture_hold_duration_requirement():
    """Verify command only triggers AFTER the configured hold duration has elapsed."""
    manager = CommandManager(hold_seconds=0.5, cooldown_seconds=1.0)

    # Time 0.0: Initial detection of NEXT_SLIDE
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0) is None

    # Time 0.3: Still holding, but 0.3s < 0.5s threshold
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.3) is None

    # Time 0.49: Just under threshold
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.49) is None

    # Time 0.51: Held for >= 0.5s -> Should trigger!
    cmd = manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.51)
    assert cmd == "NEXT_SLIDE"


def test_interrupted_gesture_resets_hold_timer():
    """If the gesture flickers or changes before hold duration, timer must reset."""
    manager = CommandManager(hold_seconds=0.5, cooldown_seconds=1.0)

    # Start holding NEXT_SLIDE
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.3) is None

    # Hand briefly flickers to neutral/unknown at 0.35
    assert manager.process_gesture(GESTURE_UNKNOWN, current_time=0.35) is None

    # Starts holding NEXT_SLIDE again at 0.40
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.40) is None

    # At 0.60, only 0.20s elapsed since restart -> Must NOT trigger
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.60) is None

    # At 0.95, 0.55s elapsed since restart -> Triggers!
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.95) == "NEXT_SLIDE"


# ==============================================================================
# TESTS FOR COOLDOWN AND ANTI-SPAM PREVENTION
# ==============================================================================
def test_command_cooldown_enforcement():
    """Verify no new command triggers during the cooldown window."""
    manager = CommandManager(hold_seconds=0.5, cooldown_seconds=1.0)

    # Trigger first command at t=0.5
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.5) == "NEXT_SLIDE"

    # User releases hand at t=0.6
    manager.process_gesture(GESTURE_UNKNOWN, current_time=0.6)

    # User immediately makes PREV_SLIDE at t=0.7
    manager.process_gesture(GESTURE_PREV_SLIDE, current_time=0.7)

    # At t=1.25, PREV_SLIDE has been held for 0.55s, but cooldown (1.0s) has NOT elapsed (last=0.5, next allowed=1.5)
    assert manager.process_gesture(GESTURE_PREV_SLIDE, current_time=1.25) is None

    # At t=1.55, cooldown has passed (0.5 + 1.0 = 1.5) -> PREV_SLIDE triggers!
    assert manager.process_gesture(GESTURE_PREV_SLIDE, current_time=1.55) == "PREV_SLIDE"


def test_prevention_of_repeated_commands_when_continuously_held():
    """Holding a gesture continuously must NOT spam multiple slide advances."""
    manager = CommandManager(
        hold_seconds=0.5,
        cooldown_seconds=1.0,
        require_neutral_release=True,
    )

    # Initial trigger at t=0.5
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.5) == "NEXT_SLIDE"

    # Continuously held through cooldown and beyond
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=1.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=1.6) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=3.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=10.0) is None

    # Now release to neutral
    manager.process_gesture(GESTURE_UNKNOWN, current_time=10.5)

    # Next hold should trigger normally after neutral release
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=11.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=11.6) == "NEXT_SLIDE"


# ==============================================================================
# TESTS FOR STATE MACHINE (ACTIVE, PAUSED, SAFE MODE)
# ==============================================================================
def test_pause_and_resume_state_transitions():
    """Test closed fist pauses controls and open palm resumes controls."""
    manager = CommandManager(hold_seconds=0.5, cooldown_seconds=0.5)

    assert manager.is_active() is True
    assert manager.is_paused() is False

    # Perform Closed Fist (PAUSE) at t=0.0 -> t=0.6
    manager.process_gesture(GESTURE_PAUSE, current_time=0.0)
    cmd = manager.process_gesture(GESTURE_PAUSE, current_time=0.6)
    assert cmd == "PAUSE"
    assert manager.is_paused() is True
    assert manager.is_active() is False

    # Release hand to neutral
    manager.process_gesture(GESTURE_UNKNOWN, current_time=1.2)

    # While paused, slide commands must be ignored
    manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=1.3)
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=2.0) is None

    # Perform Open Palm (RESUME) at t=2.5 -> t=3.1
    manager.process_gesture(GESTURE_UNKNOWN, current_time=2.2)
    manager.process_gesture(GESTURE_RESUME, current_time=2.5)
    cmd_resume = manager.process_gesture(GESTURE_RESUME, current_time=3.1)
    assert cmd_resume == "RESUME"
    assert manager.is_active() is True
    assert manager.is_paused() is False

    # Now slide commands work again
    manager.process_gesture(GESTURE_UNKNOWN, current_time=3.7)
    manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=3.8)
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=4.4) == "NEXT_SLIDE"


def test_safe_mode_toggle_and_state():
    """Verify safe mode toggling."""
    manager = CommandManager(safe_mode=False)
    assert manager.safe_mode is False
    assert manager.state == SystemState.ACTIVE

    manager.toggle_safe_mode()
    assert manager.safe_mode is True
    assert manager.state == SystemState.SAFE_MODE

    manager.toggle_safe_mode()
    assert manager.safe_mode is False
    assert manager.state == SystemState.ACTIVE


# ==============================================================================
# TESTS FOR PRESENTATION CONTROLLER WITH MOCKED PYAUTOGUI
# ==============================================================================
def test_presentation_controller_with_mock():
    """Verify PresentationController dispatches the correct keys using a mock."""
    with patch("pyautogui.press") as mock_press:
        controller = PresentationController(
            safe_mode=False,
            key_next="right",
            key_prev="left",
            enable_slideshow_keys=True,
            key_start="f5",
            key_exit="escape",
        )

        # Test Next Slide
        controller.next_slide()
        mock_press.assert_called_with("right")

        # Test Prev Slide
        controller.prev_slide()
        mock_press.assert_called_with("left")

        # Test Start Slideshow
        controller.start_slideshow()
        mock_press.assert_called_with("f5")

        # Test Exit Slideshow
        controller.exit_slideshow()
        mock_press.assert_called_with("escape")

        # Test Generic command routing
        controller.execute_command("NEXT_SLIDE")
        mock_press.assert_called_with("right")


def test_presentation_controller_safe_mode_does_not_call_os():
    """In Safe Mode, PyAutoGUI must NEVER be invoked."""
    with patch("pyautogui.press") as mock_press:
        controller = PresentationController(safe_mode=True)
        controller.next_slide()
        controller.prev_slide()
        controller.execute_command("NEXT_SLIDE")

        mock_press.assert_not_called()


def test_flicker_grace_buffer_preserves_hold_timer():
    """Verify that a sub-flicker-grace momentary dropout does not reset hold progress."""
    manager = CommandManager(
        hold_seconds=0.5,
        cooldown_seconds=1.0,
        flicker_grace_seconds=0.15,
    )

    # Start holding NEXT_SLIDE at t=0.0
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0) is None
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.3) is None

    # Momentary 0.05s flicker to UNKNOWN at t=0.35 (within 0.15s grace buffer)
    assert manager.process_gesture(GESTURE_UNKNOWN, current_time=0.35) is None

    # Returns to NEXT_SLIDE at t=0.38
    assert manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.38) is None

    # At t=0.52, total hold time since t=0.0 is >= 0.5s -> Should successfully trigger!
    cmd = manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.52)
    assert cmd == "NEXT_SLIDE"


def test_hold_progress_remains_full_when_executed():
    """Verify hold_progress remains 1.0 after command triggers until release."""
    manager = CommandManager(hold_seconds=0.4, cooldown_seconds=1.0)

    manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.0)
    info_half = manager.get_status_info(current_time=0.2)
    assert info_half["hold_progress"] == pytest.approx(0.5, abs=0.05)

    # Trigger command at t=0.45
    manager.process_gesture(GESTURE_NEXT_SLIDE, current_time=0.45)
    info_done = manager.get_status_info(current_time=0.46)
    assert info_done["hold_progress"] == 1.0


