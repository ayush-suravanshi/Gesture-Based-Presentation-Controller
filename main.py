"""
Main Application Entry Point.

Gesture-Based Presentation Controller Using Computer Vision.
Orchestrates webcam capture, hand tracking, gesture stabilization, UI overlay rendering,
and presentation slide control.
"""

import argparse
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import config
from gesture_detector import GestureDetector
from presentation_controller import PresentationController
from src.command_manager import CommandManager, SystemState


def draw_hud_overlay(
    frame: np.ndarray,
    status_info: Dict[str, Any],
    fps: float,
    debug_mode: bool = False,
) -> None:
    """
    Render a crisp, high-visibility Heads-Up Display (HUD) overlay on the video feed.
    Optimized for viewing from 2+ meters away.

    Features:
    - Adaptive scaling for 720p HD and 1080p resolutions.
    - Real-time Hand Distance Estimation indicator (~2.0m [Optimal]).
    - High-contrast, color-coded System State badge (ACTIVE / PAUSED / SAFE MODE).
    - Large Hold Progress Bar visible from standing presentation distance.
    - Last Executed Command Badge with bright trigger feedback flash.
    """
    h, w, _ = frame.shape
    # Scale factor relative to 720p height
    scale = max(0.8, min(1.6, h / 720.0))

    # --------------------------------------------------------------------------
    # 1. Top Header Banner (Semi-transparent dark backdrop)
    # --------------------------------------------------------------------------
    banner_h = int(84 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 22), -1)
    # Bottom bar for instructions
    bottom_h = int(38 * scale)
    cv2.rectangle(overlay, (0, h - bottom_h), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Title
    title_font_scale = 0.65 * scale
    cv2.putText(
        frame,
        "Gesture Presentation Controller",
        (int(16 * scale), int(28 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        title_font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # FPS counter (top right)
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (w - int(120 * scale), int(28 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * scale,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------------------------
    # 2. System State & Hand Distance Badges
    # --------------------------------------------------------------------------
    state_str = status_info["state"]
    if status_info.get("safe_mode", False):
        badge_color = config.COLOR_SAFE_MODE
        badge_text = "[ SAFE MODE ]"
    elif state_str == "ACTIVE":
        badge_color = config.COLOR_ACTIVE
        badge_text = "[ ACTIVE ]"
    else:
        badge_color = config.COLOR_PAUSED
        badge_text = "[ PAUSED ]"

    badge_font_scale = 0.68 * scale
    cv2.putText(
        frame,
        badge_text,
        (int(16 * scale), int(64 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        badge_font_scale,
        badge_color,
        2,
        cv2.LINE_AA,
    )

    # Cooldown Status Indicator
    cooldown = status_info.get("cooldown_remaining", 0.0)
    if cooldown > 0.0:
        cd_text = f"Cooldown: {cooldown:.1f}s"
        cd_color = (0, 165, 255)
    else:
        cd_text = "Cooldown: Ready"
        cd_color = (0, 220, 100)

    cv2.putText(
        frame,
        cd_text,
        (int(210 * scale), int(64 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * scale,
        cd_color,
        1,
        cv2.LINE_AA,
    )

    # Hand Distance Indicator (Crucial for 2-meter verification!)
    dist_label = status_info.get("distance_label", "No Hand")
    dist_m = status_info.get("distance_m", 0.0)
    if 1.4 <= dist_m <= 2.6:
        dist_color = (46, 204, 113)  # Bright green for optimal ~2m range
    elif 0.5 <= dist_m < 1.4:
        dist_color = (255, 200, 0)   # Cyan for close
    elif dist_m > 2.6:
        dist_color = (0, 180, 255)   # Amber for far
    else:
        dist_color = (160, 160, 160)

    dist_display = f"Dist: {dist_label}" if dist_m > 0 else "Dist: No Hand"
    cv2.putText(
        frame,
        dist_display,
        (int(410 * scale), int(64 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * scale,
        dist_color,
        2 if (1.4 <= dist_m <= 2.6) else 1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------------------------
    # 3. Gesture Recognition & Hold Progress Bar (High visibility from 2m)
    # --------------------------------------------------------------------------
    current_gesture = status_info.get("current_gesture", "NONE")
    raw_gesture = status_info.get("raw_gesture", "NONE")
    hand_present = status_info.get("hand_present", False)
    finger_states = status_info.get("finger_states", {})
    hold_progress = status_info.get("hold_progress", 0.0)

    panel_x = int(14 * scale)
    panel_y = int(98 * scale)
    panel_w = int(450 * scale)
    panel_h = int(108 * scale)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (28, 28, 32), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (65, 65, 75), 1)

    # Active gesture label
    if hand_present:
        active_gesture_name = current_gesture if current_gesture != "NONE" else raw_gesture
        if active_gesture_name == "NEXT_SLIDE":
            label_text = "Gesture: NEXT SLIDE (Index Up)"
            label_color = (0, 255, 200)
        elif active_gesture_name == "PREV_SLIDE":
            label_text = "Gesture: PREV SLIDE (Peace / 2 Up)"
            label_color = (255, 200, 0)
        elif active_gesture_name == "RESUME":
            label_text = "Gesture: RESUME (Open Palm)"
            label_color = (0, 255, 100)
        elif active_gesture_name == "PAUSE":
            label_text = "Gesture: PAUSE (Closed Fist)"
            label_color = (0, 160, 255)
        else:
            label_text = "Gesture: Neutral / Move Hand"
            label_color = (200, 200, 200)
    else:
        label_text = "Gesture: No Hand in View"
        label_color = (150, 150, 150)

    cv2.putText(
        frame,
        label_text,
        (panel_x + int(12 * scale), panel_y + int(26 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54 * scale,
        label_color,
        2 if hand_present and label_text != "Gesture: Neutral / Move Hand" else 1,
        cv2.LINE_AA,
    )

    # Real-time finger state indicators: [I:UP] [M:--] [R:--] [P:--]
    if hand_present:
        i_s = "UP" if finger_states.get("index") else "--"
        m_s = "UP" if finger_states.get("middle") else "--"
        r_s = "UP" if finger_states.get("ring") else "--"
        p_s = "UP" if finger_states.get("pinky") else "--"
        fingers_text = f"Fingers: Idx:[{i_s}] Mid:[{m_s}] Ring:[{r_s}] Pnk:[{p_s}]"
    else:
        fingers_text = "Show hand clearly in view (1m - 2.5m)"

    cv2.putText(
        frame,
        fingers_text,
        (panel_x + int(12 * scale), panel_y + int(48 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44 * scale,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    # Hold Progress Bar
    bar_x = panel_x + int(12 * scale)
    bar_y = panel_y + int(58 * scale)
    bar_w = panel_w - int(24 * scale)
    bar_h = int(14 * scale)

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 58), -1)

    # Filled bar
    fill_w = int(bar_w * hold_progress)
    bar_color = config.COLOR_ACTIVE if hold_progress >= 0.99 else (255, 180, 50)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    if hand_present and current_gesture != "NONE":
        progress_text = f"Holding: {int(hold_progress * 100)}% (Triggering...)" if hold_progress < 0.99 else "TRIGGERED!"
    elif hand_present:
        progress_text = "Hold gesture steady to trigger slide"
    else:
        progress_text = "Waiting for hand..."

    cv2.putText(
        frame,
        progress_text,
        (panel_x + int(12 * scale), panel_y + int(94 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44 * scale,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------------------------
    # 4. Last Executed Command Badge (with Trigger Visual Flash)
    # --------------------------------------------------------------------------
    last_cmd = status_info.get("last_command", "NONE")
    feedback_active = status_info.get("feedback_active", False)

    cmd_panel_w = int(320 * scale)
    cmd_panel_h = int(90 * scale)
    cmd_panel_x = w - cmd_panel_w - int(14 * scale)
    cmd_panel_y = int(98 * scale)

    panel_color = (0, 120, 220) if feedback_active else (28, 28, 32)
    border_color = config.COLOR_TRIGGERED if feedback_active else (65, 65, 75)

    cv2.rectangle(frame, (cmd_panel_x, cmd_panel_y), (cmd_panel_x + cmd_panel_w, cmd_panel_y + cmd_panel_h), panel_color, -1)
    cv2.rectangle(frame, (cmd_panel_x, cmd_panel_y), (cmd_panel_x + cmd_panel_w, cmd_panel_y + cmd_panel_h), border_color, 2 if feedback_active else 1)

    cv2.putText(
        frame,
        "Last Command:",
        (cmd_panel_x + int(14 * scale), cmd_panel_y + int(28 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * scale,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cmd_text_color = (255, 255, 255) if feedback_active else (0, 230, 255)
    cv2.putText(
        frame,
        last_cmd,
        (cmd_panel_x + int(14 * scale), cmd_panel_y + int(66 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.66 * scale,
        cmd_text_color,
        2 if feedback_active else 1,
        cv2.LINE_AA,
    )

    # Full frame visual flash when command fires
    if feedback_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), int(6 * scale))

    # --------------------------------------------------------------------------
    # 5. Bottom Instructions & Presentation Focus Reminder
    # --------------------------------------------------------------------------
    guide_text = "[Q] Quit  |  [P] Pause/Resume  |  [S] Safe Mode  |  [R] Reset"
    cv2.putText(
        frame,
        guide_text,
        (int(16 * scale), h - int(12 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    # Presentation window reminder
    cv2.putText(
        frame,
        "* Keep slides window focused!",
        (w - int(280 * scale), h - int(12 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * scale,
        (0, 210, 255),
        1,
        cv2.LINE_AA,
    )


def run_application(
    camera_index: int = config.CAMERA_INDEX,
    video_source: Optional[str] = None,
    frame_width: int = config.FRAME_WIDTH,
    frame_height: int = config.FRAME_HEIGHT,
    safe_mode: bool = config.SAFE_MODE,
    debug_mode: bool = config.DEBUG_MODE,
) -> int:
    """
    Main execution loop. Initializes camera or video file, tracking pipeline, and presentation dispatcher.

    Returns:
        int: Exit status code (0 for success, non-zero for error).
    """
    print("=" * 68)
    print("  GESTURE-BASED PRESENTATION CONTROLLER USING COMPUTER VISION")
    print("=" * 68)
    source_desc = f"Video File: {video_source}" if video_source else f"Webcam Index: {camera_index}"
    print(f"[*] Input source: {source_desc}")
    print(f"[*] Target resolution: {frame_width}x{frame_height}")
    print(f"[*] Safe mode: {'ENABLED (Simulated keypresses)' if safe_mode else 'DISABLED (Real keypresses)'}")
    print(f"[*] Gesture hold duration: {config.GESTURE_HOLD_SECONDS}s")
    print(f"[*] Command cooldown: {config.COMMAND_COOLDOWN_SECONDS}s")
    print("-" * 68)

    # --------------------------------------------------------------------------
    # Step 1: Initialize Presentation Controller
    # --------------------------------------------------------------------------
    controller = PresentationController(safe_mode=safe_mode)
    controller.check_environment()

    # --------------------------------------------------------------------------
    # Step 2: Initialize Command Manager
    # --------------------------------------------------------------------------
    command_manager = CommandManager(
        hold_seconds=config.GESTURE_HOLD_SECONDS,
        cooldown_seconds=config.COMMAND_COOLDOWN_SECONDS,
        require_neutral_release=config.REQUIRE_NEUTRAL_RELEASE,
        safe_mode=safe_mode,
        feedback_flash_seconds=config.FEEDBACK_FLASH_SECONDS,
        flicker_grace_seconds=config.FLICKER_GRACE_SECONDS,
    )

    # --------------------------------------------------------------------------
    # Step 3: Initialize Gesture Detector (MediaPipe)
    # --------------------------------------------------------------------------
    try:
        print("[*] Loading MediaPipe hand landmarker model...")
        detector = GestureDetector(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            model_path=str(config.MODEL_PATH),
        )
        print(f"[+] MediaPipe hand detector initialized successfully (Backend: {detector.tracker.backend.upper()})")
    except Exception as err:
        print(f"\n[ERROR] Failed to initialize MediaPipe detector: {err}\n")
        print("Please check that the model asset exists at:", config.MODEL_PATH)
        return 1

    # --------------------------------------------------------------------------
    # Step 4: Open Webcam Video Capture or Video File
    # --------------------------------------------------------------------------
    capture_target = video_source if video_source else camera_index
    if video_source:
        print(f"[*] Opening video stream from file: {video_source}...")
    else:
        print(f"[*] Opening webcam device at index {camera_index}...")

    cap = cv2.VideoCapture(capture_target)

    # Set camera resolution (for webcam streams)
    if not video_source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("\n" + "!" * 68)
        if video_source:
            print(f"[ERROR] Could not open video file: {video_source}")
            print("Please verify the file path and that the video format is supported.")
        else:
            print(f"[ERROR] Could not open webcam at index {camera_index}.")
            print("Possible causes:")
            print("  1. No webcam is connected to this computer.")
            print("  2. macOS Camera permissions have not been granted for this app.")
            print("     Fix: System Settings -> Privacy & Security -> Camera -> Enable Terminal/IDE.")
            print("  3. An external camera is plugged in at a different index (e.g. --camera 1).")
        print("!" * 68 + "\n")
        detector.close()
        return 2

    source_label = f"Video file: {video_source}" if video_source else "Webcam stream"
    print(f"[+] {source_label} started successfully!")
    print("\n" + "=" * 68)
    print("  GESTURE CONTROLS GUIDE:")
    print("  - Index Finger UP       -> Next Slide (Right Arrow)")
    print("  - Two Fingers (V-Sign)  -> Previous Slide (Left Arrow)")
    print("  - Open Palm             -> Resume / Enable Gesture Control")
    print("  - Closed Fist           -> Pause Gesture Control")
    print("\n  KEYBOARD SHORTCUTS:")
    print("  - [Q] : Quit the application")
    print("  - [P] : Pause / Resume gesture control")
    print("  - [S] : Toggle Safe Mode (test without slide changes)")
    print("  - [R] : Reset command timers and cooldown")
    print("  - [D] : Toggle debug metrics")
    print("=" * 68 + "\n")
    print("[REMINDER] Please open your PowerPoint/Keynote slides and click on the slides window!")

    # --------------------------------------------------------------------------
    # Step 5: Continuous Video Frame Processing Loop
    # --------------------------------------------------------------------------
    prev_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Failed to receive frame from webcam (camera disconnected?). Exiting loop.")
                break

            current_time = time.time()
            # Calculate rolling FPS
            time_delta = current_time - prev_frame_time
            if time_delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_delta)
            prev_frame_time = current_time

            # Mirror feed horizontally for intuitive user experience
            if config.MIRROR_DISPLAY:
                frame = cv2.flip(frame, 1)

            # Process hand tracking and gesture recognition
            frame, gesture_info = detector.process_frame(frame, draw=True)
            recognized_gesture = gesture_info["gesture"]

            # Evaluate gesture against stabilization and cooldown
            dispatched_command = command_manager.process_gesture(
                recognized_gesture,
                current_time=current_time,
            )

            # If a presentation command triggered, dispatch it!
            if dispatched_command is not None:
                controller.execute_command(dispatched_command)

            # Get UI telemetry and attach distance & finger metrics for HUD display
            status_info = command_manager.get_status_info(current_time=current_time)
            status_info["distance_m"] = gesture_info.get("distance_m", 0.0)
            status_info["distance_label"] = gesture_info.get("distance_label", "No Hand")
            status_info["finger_states"] = gesture_info.get("finger_states", {})
            status_info["hand_present"] = gesture_info.get("hand_present", False)
            status_info["raw_gesture"] = gesture_info.get("raw_gesture", "NONE")
            status_info["gesture_display"] = gesture_info.get("display_name", "")
            draw_hud_overlay(frame, status_info, fps, debug_mode=debug_mode)

            # Display processed frame
            cv2.imshow(config.WINDOW_TITLE, frame)

            # Listen for keyboard hotkeys (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc to quit
                print("\n[*] Quit signal received. Closing presentation controller...")
                break
            elif key in (ord("p"), ord("P")):
                new_state = command_manager.toggle_pause()
                print(f"[*] State toggled: {new_state.value}")
            elif key in (ord("s"), ord("S")):
                new_safe = command_manager.toggle_safe_mode()
                controller.safe_mode = new_safe
                print(f"[*] Safe Mode: {'ENABLED' if new_safe else 'DISABLED'}")
            elif key in (ord("r"), ord("R")):
                command_manager.reset_cooldown()
                print("[*] Timers and cooldown reset.")
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
                print(f"[*] Debug mode: {'ENABLED' if debug_mode else 'DISABLED'}")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user (Ctrl+C). Cleaning up...")
    finally:
        # Cleanup resources cleanly
        print("[*] Releasing webcam and destroying windows...")
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[+] Application terminated cleanly.")

    return 0


def main() -> None:
    """CLI Argument parser and entry point."""
    parser = argparse.ArgumentParser(
        description="Gesture-Based Presentation Controller Using Computer Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Webcam device index (0 for default built-in camera)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an optional video file input (for testing without a live webcam)",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=config.SAFE_MODE,
        help="Run in Safe Mode (gesture detection without sending OS keypresses)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.DEBUG_MODE,
        help="Show detailed debug metrics and logging",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.FRAME_WIDTH,
        help="Capture frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.FRAME_HEIGHT,
        help="Capture frame height",
    )

    args = parser.parse_args()

    exit_code = run_application(
        camera_index=args.camera,
        video_source=args.video,
        frame_width=args.width,
        frame_height=args.height,
        safe_mode=args.safe_mode,
        debug_mode=args.debug,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

