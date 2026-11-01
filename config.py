"""
Configuration Module for Gesture-Based Presentation Controller.

This file centralizes all tunable parameters, including camera properties,
MediaPipe detector options, gesture stabilization thresholds, cooldown timings,
and presentation keyboard mappings.
"""

from pathlib import Path

# ==============================================================================
# 1. FILE SYSTEM & ASSETS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"

# ==============================================================================
# 2. CAMERA & VIDEO DISPLAY SETTINGS
# ==============================================================================
# Camera device index. 0 is typically the built-in webcam (FaceTime HD camera).
# External USB or Continuity cameras usually appear at index 1 or higher.
CAMERA_INDEX = 0

# Video capture resolution. 1280x720 (720p HD) provides more than 2.25x the pixel
# density of 480p, allowing the palm detector to reliably detect small hands at 2+ meters
# while remaining lightweight on MacBook Air (Apple Silicon Metal GPU).
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Flip the camera feed horizontally for an intuitive mirror view
MIRROR_DISPLAY = True

# Window title for OpenCV preview window
WINDOW_TITLE = "Gesture-Based Presentation Controller"

# ==============================================================================
# 3. HAND TRACKING (MEDIAPIPE) & 2-METER DISTANCE TUNING
# ==============================================================================
# Maximum number of hands to detect concurrently (1 is standard for presentations)
MAX_NUM_HANDS = 1

# Detection and tracking confidence thresholds (0.0 to 1.0).
# Tuned to 0.45 / 0.40 so small, distant hands at 2 meters are detected reliably
# without dropping tracking during hand movement.
MIN_DETECTION_CONFIDENCE = 0.45
MIN_TRACKING_CONFIDENCE = 0.40

# Temporal landmark smoothing (Exponential Moving Average) to eliminate sub-pixel
# landmark jitter on distant hands.
ENABLE_LANDMARK_SMOOTHING = True
LANDMARK_SMOOTHING_ALPHA = 0.65  # Weight for new frame (0.65 = responsive + stable)

# Rolling history window (in frames) for gesture majority voting.
# Fast 3-frame buffer avoids latency when presenting while absorbing momentary jitter.
GESTURE_HISTORY_FRAMES = 3

# Long-Range Mode: enables enhanced anatomical ratio calculations and large HUD badges
LONG_RANGE_MODE = True

# ==============================================================================
# 4. GESTURE STABILIZATION & TIMING (CRITICAL ANTI-ACCIDENTAL CHANGING)
# ==============================================================================
# Duration (in seconds) a user must continuously hold a recognized gesture
# before the corresponding command is triggered. 0.38s is fast, responsive, and fatigue-free.
GESTURE_HOLD_SECONDS = 0.38

# Cooldown duration (in seconds) after a command fires, during which no new
# commands can be triggered. Prevents rapid unintended slide skipping.
COMMAND_COOLDOWN_SECONDS = 0.8

# Flicker grace period (in seconds): momentary single-frame drops will not wipe hold progress
FLICKER_GRACE_SECONDS = 0.15

# If True, the user MUST return their hand to a neutral state (or release the gesture)
# before the same command can be executed again. Holding a gesture indefinitely
# will not spam slide changes.
REQUIRE_NEUTRAL_RELEASE = True

# ==============================================================================
# 5. PRESENTATION CONTROLLER & KEYBOARD SHORTCUTS
# ==============================================================================
# Safe Mode: When True, gestures are recognized and logged on-screen,
# but NO actual keyboard key presses are dispatched to the operating system.
# Highly recommended when testing or presenting without a slideshow open.
SAFE_MODE = False

# Keyboard shortcuts sent via PyAutoGUI.
# Compatible with Microsoft PowerPoint, Google Slides, Apple Keynote, and PDF viewers.
KEY_NEXT_SLIDE = "right"
KEY_PREV_SLIDE = "left"

# Optional presentation controls (only active if ENABLE_SLIDESHOW_SHORTCUTS is True)
ENABLE_SLIDESHOW_SHORTCUTS = False
KEY_START_SLIDESHOW = "f5"
KEY_EXIT_SLIDESHOW = "escape"

# Small delay (in seconds) after sending a PyAutoGUI key press to ensure OS handles it
KEY_PRESS_INTERVAL = 0.05

# ==============================================================================
# 6. ON-SCREEN UI & VISUAL STYLING (BGR FORMAT FOR OPENCV)
# ==============================================================================
COLOR_ACTIVE = (46, 204, 113)     # Emerald Green: normal active operation
COLOR_PAUSED = (0, 140, 255)     # Deep Orange: gesture control paused
COLOR_SAFE_MODE = (255, 191, 0)  # Cyan / Deep Sky Blue: safe mode active
COLOR_NEUTRAL = (180, 180, 180)  # Light Gray: waiting / neutral
COLOR_TRIGGERED = (0, 255, 255)  # Bright Yellow: command fired flash

# Command feedback flash duration in seconds on the HUD
FEEDBACK_FLASH_SECONDS = 0.35

# ==============================================================================
# 7. DEBUGGING & LOGGING
# ==============================================================================
# If True, prints extra diagnostic data and draws detailed coordinate metrics
DEBUG_MODE = False

