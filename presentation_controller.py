"""
Presentation Controller Module.

Dispatches presentation navigation keyboard shortcuts using PyAutoGUI.
Handles safe mode, key press rate limits, and macOS Accessibility permission exceptions.
"""

import logging
import sys
from typing import Optional

try:
    import pyautogui
    # Disable PyAutoGUI pause delay for snappy response
    pyautogui.PAUSE = 0.02
    # Keep FAILSAFE enabled (moving mouse to corner raises exception)
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None

import config

logger = logging.getLogger(__name__)


class PresentationController:
    """
    Interfaces between high-level presentation commands and the operating
    system keyboard event queue using PyAutoGUI.
    """

    def __init__(
        self,
        safe_mode: bool = config.SAFE_MODE,
        key_next: str = config.KEY_NEXT_SLIDE,
        key_prev: str = config.KEY_PREV_SLIDE,
        enable_slideshow_keys: bool = config.ENABLE_SLIDESHOW_SHORTCUTS,
        key_start: str = config.KEY_START_SLIDESHOW,
        key_exit: str = config.KEY_EXIT_SLIDESHOW,
    ) -> None:
        self.safe_mode = safe_mode
        self.key_next = key_next
        self.key_prev = key_prev
        self.enable_slideshow_keys = enable_slideshow_keys
        self.key_start = key_start
        self.key_exit = key_exit

        self._has_accessibility_warning_printed = False

    def check_environment(self) -> None:
        """
        Verify PyAutoGUI availability and remind user about macOS Accessibility permissions.
        """
        if pyautogui is None:
            print(
                "\n[WARNING] PyAutoGUI is not installed! Running in forced Safe Mode.\n"
                "Install PyAutoGUI via: pip install pyautogui\n"
            )
            self.safe_mode = True
            return

        if sys.platform == "darwin":
            print(
                "\n"
                "====================================================================\n"
                "  macOS ACCESSIBILITY PERMISSION NOTICE                             \n"
                "====================================================================\n"
                "  To send arrow keys to PowerPoint, Keynote, or Google Slides:     \n"
                "  macOS requires Accessibility permissions for your Terminal / IDE.\n"
                "\n"
                "  If keystrokes do not trigger slide changes:                      \n"
                "  1. Open System Settings > Privacy & Security > Accessibility.     \n"
                "  2. Enable the toggle for your Terminal (Terminal, iTerm, or Code).\n"
                "====================================================================\n"
            )

    def execute_command(self, command_name: str) -> bool:
        """
        Route and execute a discrete command string.

        Args:
            command_name: 'NEXT_SLIDE', 'PREV_SLIDE', 'START_SLIDESHOW', 'EXIT_SLIDESHOW'.

        Returns:
            bool: True if command was dispatched (or simulated in safe mode); False otherwise.
        """
        if command_name == "NEXT_SLIDE":
            return self.next_slide()
        elif command_name == "PREV_SLIDE":
            return self.prev_slide()
        elif command_name == "START_SLIDESHOW":
            return self.start_slideshow()
        elif command_name == "EXIT_SLIDESHOW":
            return self.exit_slideshow()
        elif command_name in ("PAUSE", "RESUME"):
            # Internal state commands do not send keyboard strokes
            return True
        return False

    def next_slide(self) -> bool:
        """Send next slide keyboard shortcut."""
        return self._send_key(self.key_next, "Next Slide")

    def prev_slide(self) -> bool:
        """Send previous slide keyboard shortcut."""
        return self._send_key(self.key_prev, "Previous Slide")

    def start_slideshow(self) -> bool:
        """Send start slideshow shortcut if enabled."""
        if not self.enable_slideshow_keys:
            print("[INFO] Start slideshow shortcut is disabled in config.py.")
            return False
        return self._send_key(self.key_start, "Start Slideshow")

    def exit_slideshow(self) -> bool:
        """Send exit slideshow shortcut if enabled."""
        if not self.enable_slideshow_keys:
            print("[INFO] Exit slideshow shortcut is disabled in config.py.")
            return False
        return self._send_key(self.key_exit, "Exit Slideshow")

    def _send_key(self, key_name: str, action_label: str) -> bool:
        """
        Internal worker to press a keyboard key safely.
        """
        if self.safe_mode:
            print(f"[SAFE MODE] Simulated Key Dispatched: {key_name.upper()} ({action_label})")
            return True

        if pyautogui is None:
            print(f"[ERROR] Cannot send key '{key_name}': PyAutoGUI not installed.")
            return False

        try:
            pyautogui.press(key_name)
            print(f"[COMMAND] Dispatched Key: {key_name.upper()} ({action_label})")
            return True
        except Exception as err:
            self._handle_permission_error(err)
            return False

    def _handle_permission_error(self, err: Exception) -> None:
        """Handle and explain macOS Accessibility or OS permission errors."""
        print(f"\n[ERROR] Failed to send keystroke: {err}")
        if sys.platform == "darwin" and not self._has_accessibility_warning_printed:
            print(
                "\n"
                "--------------------------------------------------------------------\n"
                "[!] macOS Accessibility Permission Missing:                          \n"
                "    PyAutoGUI was blocked by macOS security policy.                 \n"
                "    To fix this:                                                    \n"
                "    1. Open System Settings -> Privacy & Security -> Accessibility. \n"
                "    2. Add/Toggle ON your Terminal application or Python runner.    \n"
                "    3. Restart the application.                                     \n"
                "--------------------------------------------------------------------\n"
            )
            self._has_accessibility_warning_printed = True

