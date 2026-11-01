"""
Command Manager Module.

Implements gesture stabilization, hold timing, anti-repeat logic, cooldown enforcement,
and state machine transitions (Active, Paused, Safe Mode).
"""

from enum import Enum
import time
from typing import Any, Callable, Dict, Optional

from .gesture_utils import (
    GESTURE_NEXT_SLIDE,
    GESTURE_PAUSE,
    GESTURE_PREV_SLIDE,
    GESTURE_RESUME,
    GESTURE_UNKNOWN,
)


class SystemState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    SAFE_MODE = "SAFE_MODE"


class CommandManager:
    """
    Manages the lifecycle of gesture events and converts stable hand gestures
    into discrete presentation commands.

    Key Safety Features:
    1. Stabilization: Requires holding a gesture continuously for `hold_seconds`.
    2. Cooldown: Enforces a pause (`cooldown_seconds`) between consecutive commands.
    3. Neutral Release: Prevents endless triggering while holding the same gesture.
    4. State Machine: Handles Paused vs Active state and Safe Mode.
    """

    def __init__(
        self,
        hold_seconds: float = 0.5,
        cooldown_seconds: float = 1.0,
        require_neutral_release: bool = True,
        safe_mode: bool = False,
        feedback_flash_seconds: float = 0.35,
        flicker_grace_seconds: float = 0.0,
    ) -> None:
        self.hold_seconds = float(hold_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.require_neutral_release = require_neutral_release
        self.safe_mode = safe_mode
        self.feedback_flash_seconds = feedback_flash_seconds
        self.flicker_grace_seconds = float(flicker_grace_seconds)

        # System state: start active (or safe mode if specified)
        self.state: SystemState = SystemState.SAFE_MODE if safe_mode else SystemState.ACTIVE

        # Tracking state
        self._current_gesture: Optional[str] = None
        self._gesture_start_time: float = 0.0
        self._unknown_start_time: float = 0.0
        self._last_command_time: float = -999999.0
        self._last_command_name: Optional[str] = None
        self._executed_for_current_hold: bool = False
        self._neutral_released: bool = True

    # --------------------------------------------------------------------------
    # State Control Properties & Methods
    # --------------------------------------------------------------------------
    def is_active(self) -> bool:
        """Check if gesture commands are actively being processed."""
        return self.state in (SystemState.ACTIVE, SystemState.SAFE_MODE)

    def is_paused(self) -> bool:
        """Check if gesture controls are currently paused."""
        return self.state == SystemState.PAUSED

    def toggle_pause(self) -> SystemState:
        """Manually toggle between ACTIVE and PAUSED states."""
        if self.state == SystemState.PAUSED:
            self.state = SystemState.SAFE_MODE if self.safe_mode else SystemState.ACTIVE
            self._last_command_name = "RESUMED (MANUAL)"
        else:
            self.state = SystemState.PAUSED
            self._last_command_name = "PAUSED (MANUAL)"
        self._last_command_time = time.time()
        return self.state

    def toggle_safe_mode(self) -> bool:
        """Toggle safe mode (prevents real OS keyboard execution)."""
        self.safe_mode = not self.safe_mode
        if self.state != SystemState.PAUSED:
            self.state = SystemState.SAFE_MODE if self.safe_mode else SystemState.ACTIVE
        return self.safe_mode

    def reset_cooldown(self) -> None:
        """Immediately clear active cooldown and hold timers."""
        self._last_command_time = -999999.0
        self._gesture_start_time = 0.0
        self._current_gesture = None
        self._executed_for_current_hold = False
        self._neutral_released = True

    # --------------------------------------------------------------------------
    # Core Processing Loop
    # --------------------------------------------------------------------------
    def process_gesture(
        self,
        gesture: str,
        current_time: Optional[float] = None,
    ) -> Optional[str]:
        """
        Evaluate a recognized gesture against stability and cooldown criteria.

        Args:
            gesture: The raw recognized gesture string.
            current_time: Optional timestamp (defaults to time.time()).

        Returns:
            Optional[str]: The command to execute (e.g. 'NEXT_SLIDE', 'PREV_SLIDE',
                           'PAUSE', 'RESUME'), or None if criteria not yet met.
        """
        now = time.time() if current_time is None else current_time

        # Handle Neutral / Unknown Gestures or No Hand
        if gesture == GESTURE_UNKNOWN:
            if self._current_gesture is not None and self.flicker_grace_seconds > 0.0:
                if self._unknown_start_time == 0.0:
                    self._unknown_start_time = now
                # If still within flicker grace duration, do not reset hold yet
                if (now - self._unknown_start_time) < self.flicker_grace_seconds:
                    return None

            # Grace period expired or flicker grace is 0.0 -> Reset
            self._current_gesture = None
            self._gesture_start_time = 0.0
            self._unknown_start_time = 0.0
            self._executed_for_current_hold = False
            self._neutral_released = True
            return None

        # When a valid gesture is confirmed, clear any active unknown flicker timer
        self._unknown_start_time = 0.0

        # Detect gesture switch
        if gesture != self._current_gesture:
            self._current_gesture = gesture
            self._gesture_start_time = now
            self._executed_for_current_hold = False
            # Switching to a different gesture satisfies release of prior gesture
            self._neutral_released = True

        # Check if already executed during this uninterrupted gesture hold
        if self.require_neutral_release and self._executed_for_current_hold:
            return None

        # Check stabilization / hold duration
        elapsed_hold = now - self._gesture_start_time
        if elapsed_hold < self.hold_seconds:
            return None

        # Check cooldown period
        elapsed_cooldown = now - self._last_command_time
        if elapsed_cooldown < self.cooldown_seconds:
            return None

        # At this point, gesture has been held long enough and cooldown has elapsed.
        command_to_trigger: Optional[str] = None

        if gesture == GESTURE_PAUSE:
            # Closed Fist: Pause controls
            self.state = SystemState.PAUSED
            command_to_trigger = "PAUSE"

        elif gesture == GESTURE_RESUME:
            # Open Palm: Resume controls
            self.state = SystemState.SAFE_MODE if self.safe_mode else SystemState.ACTIVE
            command_to_trigger = "RESUME"

        elif gesture == GESTURE_NEXT_SLIDE:
            # Only trigger slide commands when ACTIVE or SAFE_MODE (not when PAUSED)
            if self.is_active():
                command_to_trigger = "NEXT_SLIDE"

        elif gesture == GESTURE_PREV_SLIDE:
            if self.is_active():
                command_to_trigger = "PREV_SLIDE"

        if command_to_trigger is not None:
            self._last_command_name = command_to_trigger
            self._last_command_time = now
            self._executed_for_current_hold = True
            self._neutral_released = False
            return command_to_trigger

        return None

    # --------------------------------------------------------------------------
    # Telemetry & Status Reporting for UI
    # --------------------------------------------------------------------------
    def get_status_info(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute real-time metrics for on-screen HUD rendering.

        Returns:
            dict containing state, hold progress (0.0 to 1.0), cooldown remaining,
            feedback flash status, and last executed command.
        """
        now = time.time() if current_time is None else current_time

        # Calculate hold progress (0.0 -> 1.0)
        hold_progress = 0.0
        if self._current_gesture is not None:
            if self._executed_for_current_hold:
                hold_progress = 1.0
            else:
                elapsed = now - self._gesture_start_time
                hold_progress = max(0.0, min(1.0, elapsed / max(self.hold_seconds, 0.001)))

        # Calculate cooldown remaining (in seconds)
        time_since_last = now - self._last_command_time
        cooldown_remaining = max(0.0, self.cooldown_seconds - time_since_last)
        is_ready = cooldown_remaining <= 0.0

        # Command trigger visual flash
        feedback_active = time_since_last <= self.feedback_flash_seconds

        return {
            "state": self.state.value,
            "is_active": self.is_active(),
            "is_paused": self.is_paused(),
            "safe_mode": self.safe_mode,
            "current_gesture": self._current_gesture or "NONE",
            "hold_progress": hold_progress,
            "cooldown_remaining": cooldown_remaining,
            "is_ready": is_ready,
            "last_command": self._last_command_name or "NONE",
            "feedback_active": feedback_active,
            "neutral_released": self._neutral_released,
        }

