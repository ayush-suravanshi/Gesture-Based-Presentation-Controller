# Gesture-Based Presentation Controller Using Computer Vision 🖐️📽️

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-27%20Passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon%20%26%20Intel)-lightgrey.svg)]()

A modular, real-time computer vision application that enables seamless touchless slide navigation for presentations (PowerPoint, Keynote, Google Slides, PDF decks) using hand gestures captured through a standard webcam.

Designed specifically for **AIML engineering students**, college project demonstrations, and technical portfolio showcases.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Hand Gestures & Controls](#-hand-gestures--controls)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [macOS Accessibility Permissions Setup](#-macos-accessibility-permissions-setup)
- [Running the Application](#-running-the-application)
- [Connecting to Presentation Apps](#-connecting-to-presentation-apps)
- [Testing](#-testing)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [College Viva & Demonstration Guide](#-college-viva--demonstration-guide)
- [Known Limitations & Future Improvements](#-known-limitations--future-improvements)

---

## 🎯 Overview & Architecture

Traditional slide clickers can be misplaced, run out of battery, or restrict natural body language. This system uses an onboard webcam and state-of-the-art deep learning landmark estimation to transform human hand poses into reliable presentation commands.

### Core Engineering Highlights:
- **Zero Heavy Neural Network Retraining Required**: Employs Google MediaPipe's single-stage hand landmark pipeline detecting 21 3D joints per hand at 30+ FPS on Apple Silicon.
- **2-Meter Long-Range Detection**: Specially calibrated for standing presenters at 2 meters (6.5 ft) using 720p HD resolution, distance-tuned confidence thresholds (0.45/0.40), Exponential Moving Average (EMA) landmark smoothing, and anatomical knuckle-to-tip ratios.
- **Live Distance Estimation**: Anthropometric palm-scale measurement displaying real-time presenter distance right on the HUD (e.g. `Dist: 2.0m [Optimal 2m]`).
- **Dual-Backend Support**: Seamlessly supports both the latest modern **MediaPipe Tasks Vision API** (`HandLandmarker`) and legacy `mp.solutions.hands`.
- **Accidental Trigger Prevention (Anti-Spam)**: Implements temporal gesture stabilization, hold confirmation buffers, command cooldown timers, and a mandatory neutral release state machine.
- **Full On-Screen HUD Overlay**: Semi-transparent OpenCV Heads-Up Display featuring live FPS, color-coded system state (`ACTIVE` / `PAUSED` / `SAFE MODE`), real-time hold progress bar, cooldown countdown, distance badge, and visual trigger feedback flash.

---

## 🖐️ Hand Gestures & Controls

| Gesture | Hand Pose | Finger States | Slide Action | Keyboard Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Next Slide** | ☝️ **Index Pointing** | Index **UP**<br>Middle, Ring, Pinky **DOWN** | Advance to next slide | `Right Arrow` (`→`) |
| **Previous Slide** | ✌️ **Peace / V-Sign** | Index & Middle **UP**<br>Ring, Pinky **DOWN** | Return to previous slide | `Left Arrow` (`←`) |
| **Resume Control** | 🖐️ **Open Palm** | All 4–5 fingers **UP** | Re-enable / Resume gesture control | State -> `ACTIVE` |
| **Pause Control** | ✊ **Closed Fist** | All fingers **DOWN** | Freeze slide changes (prevents misfires) | State -> `PAUSED` |

### On-Screen Keyboard Hotkeys:
- **`Q` or `Esc`**: Safely exit the application and release camera resources.
- **`P`**: Manually toggle between `ACTIVE` and `PAUSED` states.
- **`S`**: Toggle **Safe Mode** (tests gestures and displays HUD without sending keystrokes).
- **`R`**: Reset active cooldown and hold timers immediately.
- **`D`**: Toggle debug diagnostics (coordinates, raw finger metrics).

---

## 🧠 Anti-Accidental Slide Change Mechanism

A major challenge in vision-based gesture control is preventing accidental slide skipping while speaking or naturally moving hands.

This application implements a **four-stage defense**:

```
      [ Raw Video Frame ]
              │
              ▼
   [ Hand Landmark Estimation ]
              │
              ▼
    [ Gesture Classification ]
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 1. Stabilization: Hold Gesture for 0.5 seconds?        │
   │    └── NO  ─► Accumulate time & update progress bar    │
   │    └── YES ─► Proceed to Stage 2                       │
   │                                                        │
   │ 2. Cooldown: Has 1.0 second elapsed since last action? │
   │    └── NO  ─► Block command dispatch                   │
   │    └── YES ─► Proceed to Stage 3                       │
   │                                                        │
   │ 3. Neutral Release: Has user reset to neutral/relaxed? │
   │    └── NO  ─► Block repeated firing while held         │
   │    └── YES ─► Proceed to Stage 4                       │
   │                                                        │
   │ 4. System State: Is system ACTIVE (not PAUSED)?        │
   │    └── YES ─► Dispatch Keyboard Key via PyAutoGUI      │
   │    └── NO  ─► Ignore slide movements                   │
   └────────────────────────────────────────────────────────┘
```

1. **Activation Duration (`GESTURE_HOLD_SECONDS = 0.38s`)**: Fast and responsive, requiring you to hold the pose for just over a third of a second. Transient hand gestures or hand talking won't accidentally flip slides.
2. **Command Cooldown (`COMMAND_COOLDOWN_SECONDS = 0.8s`)**: After a slide changes, no new slide command can fire for 0.8 seconds, preventing accidental double-skips.
3. **Flicker Grace Buffer (`FLICKER_GRACE_SECONDS = 0.15s`)**: If a single video frame flickers or drops at 2 meters, the hold progress bar does not instantly reset to 0%.
4. **Neutral Release Requirement (`REQUIRE_NEUTRAL_RELEASE = True`)**: If you keep holding your index finger up for 10 seconds, **it will only advance the slide once**. To advance again, you must release to neutral and re-gesture.
5. **Pause Lock (`Closed Fist`)**: During Q&A or discussions, make a closed fist to pause recognition so hand gestures do not trigger slides. Show an open palm to resume!

---

## 📏 2-Meter Long-Range Detection Architecture

Standard computer vision tutorials fail when the presenter stands 2 meters away from their laptop because the hand appears 4x smaller in pixel area, causing palm detection failure, landmark jitter, and frequent dropped frames.

This application solves 2-meter detection through six coordinated optimizations:

### 1. High-Density Resolution (720p HD / 1080p Full HD)
- Default resolution is set to **1280x720** (with optional 1920x1080).
- At 2m, a hand occupies $\approx 70 - 100$ pixels across in 720p (compared to only $\approx 30$ pixels in 480p). This provides the single-shot palm detector with enough edge gradient information to detect hands consistently.

### 2. Distance-Tuned Confidence Thresholds
- Lowered palm detector trigger threshold from `0.65` to `0.45` and tracking confidence to `0.40`.
- Distant hands inherently have lower raw confidence scores due to distance perspective. The calibrated threshold prevents MediaPipe from discarding valid hands at 2m.

### 3. Landmark Exponential Moving Average (EMA) Smoothing
- Small hands at 2m suffer from sub-pixel coordinate fluctuations (Gaussian landmark jitter).
- Implemented `LandmarkSmoother`:
  $$\hat{L}_t = \alpha \cdot L_t + (1 - \alpha) \cdot \hat{L}_{t-1} \quad (\alpha = 0.65)$$
  This eliminates high-frequency trembling on distant fingertips while preserving instant physical responsiveness.

### 4. Kinematic Straightness Ratio & Anatomical Metrics
- Standing presenters naturally hold their hand at chest height, often tilting their palm forward towards the camera.
- Rather than relying on simple 2D $y$-coordinate elevation (`tip.y < pip.y`), the algorithm evaluates kinematic straightness and phalanx ratios:
  $$\text{Straightness} = \frac{\text{dist}(\text{MCP}, \text{TIP})}{\text{dist}(\text{MCP}, \text{PIP}) + \text{dist}(\text{PIP}, \text{DIP}) + \text{dist}(\text{DIP}, \text{TIP})} \ge 0.70$$
  This ratio accurately separates an extended finger ($\ge 0.85$) from a curled finger ($< 0.55$), eliminating false detections when holding an index finger up.

### 5. Rolling-History Majority Voting & Flicker Grace
- At 2 meters, single video frames occasionally flicker.
- `GestureRecognizer` maintains a rolling 3-frame history window with majority voting, combined with a `0.15s` flicker grace buffer in `CommandManager`, ensuring smooth and uninterrupted gesture holds.

### 6. Live Distance & Real-Time Finger Indicators on HUD
- Displays anthropometric palm-span distance estimation right on the video feed:
  - `< 1.0m`: `[Close]` (Cyan)
  - `1.4m - 2.5m`: `[Optimal 2m]` (Bright Green)
  - `> 2.6m`: `[Far]` (Amber)
- Real-time finger state indicators (`Fingers: Idx:[UP] Mid:[--] Ring:[--] Pnk:[--]`) show exactly which fingers are currently recognized as extended or folded.

---

## 🏗️ Project Structure

```
GesturePresentationController/
│
├── main.py                      # Application entry point & OpenCV display loop
├── config.py                    # Central configuration (camera, timings, keys)
├── gesture_detector.py          # Unified pipeline connecting tracking and recognition
├── gesture_recognizer.py        # High-level gesture classification & UI labels
├── presentation_controller.py   # PyAutoGUI keyboard dispatcher & permission guards
├── requirements.txt             # Dependency definitions
├── pytest.ini                   # Pytest path and runner configuration
├── README.md                    # Project documentation & viva guide
├── .gitignore                   # Version control ignore rules
│
├── src/
│   ├── __init__.py              # Source package initializer
│   ├── hand_tracker.py          # MediaPipe 21-landmark tracking & skeleton drawing
│   ├── gesture_utils.py         # Geometric math, distance checks, finger states
│   └── command_manager.py       # State machine, stabilization, anti-repeat logic
│
├── tests/
│   ├── conftest.py              # Pytest environment fixtures
│   ├── test_gesture_utils.py    # Unit tests for geometric & finger calculations
│   ├── test_command_manager.py  # Unit tests for hold timings, cooldown, mocks
│   └── test_gesture_recognizer.py# Unit tests for recognizer classifications
│
└── assets/
    └── hand_landmarker.task     # Google MediaPipe Hand Landmarker model bundle
```

---

## 💻 System Requirements

- **Operating System**: macOS 11.0+ (Big Sur, Monterey, Ventura, Sonoma, Sequoia)
  - Native support for **Apple Silicon (M1, M2, M3, M4)** and Intel Macs.
- **Python**: Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Hardware**: Built-in FaceTime HD camera or any USB/Continuity webcam.
- **RAM**: 4 GB minimum (approx. 180 MB memory footprint during execution).

---

## 🚀 Installation & Setup

Follow these exact steps in your macOS Terminal:

### 1. Clone or Navigate to the Project Directory
```bash
cd "/Users/ayushshaileshsuravanshi/Desktop/Cv project/GesturePresentationController"
```

### 2. Create an Isolated Virtual Environment
```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment
```bash
source .venv/bin/activate
```
*(Your terminal prompt should now be prefixed with `(.venv)`).*

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🔐 macOS Accessibility Permissions Setup

On macOS, security policies restrict background scripts from dispatching synthetic keystrokes to other applications unless **Accessibility** access is granted.

If keystrokes do not advance your slides:
1. Open **System Settings** on your Mac.
2. Navigate to **Privacy & Security** > **Accessibility**.
3. Click the **`+`** icon (or toggle the switch).
4. Add and enable your terminal application:
   - **Terminal** (if using standard macOS Terminal)
   - **iTerm2** (if using iTerm)
   - **Visual Studio Code** (if running inside VS Code terminal)
5. Also verify Camera access under **Privacy & Security** > **Camera**.

> **Pro Tip**: Use `--safe-mode` anytime you want to test gesture tracking without sending actual keypresses to your computer.

---

## 🎮 Running the Application

### Standard Mode (Active Slide Control)
```bash
python main.py
```

### Safe Mode (HUD & Tracking without sending Keystrokes)
```bash
python main.py --safe-mode
```

### External Webcam Index
If using an external USB camera or iPhone Continuity Camera:
```bash
python main.py --camera 1
```

### Debug Diagnostics Mode
```bash
python main.py --debug
```

### Offline Video File Testing
To demonstrate the system without a live camera (e.g. recorded video):
```bash
python main.py --video path/to/demo_video.mp4
```

---

## 🖥️ Connecting to Presentation Apps

This project works out-of-the-box with all major presentation tools:

1. **Microsoft PowerPoint** (`.pptx`)
2. **Apple Keynote**
3. **Google Slides** (in Google Chrome, Safari, or Arc)
4. **Adobe Acrobat Reader / Preview** (PDF presentation mode)
5. **Canva / Pitch / Slidev**

### How to use during a presentation:
1. Launch `python main.py`.
2. Open your PowerPoint or Google Slides presentation.
3. Enter slideshow mode (`F5` or `Cmd + Shift + Enter`).
4. **Important**: Click once on the slideshow window so it has active system focus.
5. Stand or sit at a comfortable distance (1–2.5 meters from webcam).
6. Raise your **Index Finger** to go forward.
7. Raise **Index + Middle Fingers** (V-sign) to go back.
8. Make a **Closed Fist** when pausing or answering questions.
9. Show an **Open Palm** when ready to resume!

---

## 🧪 Testing

The project includes comprehensive unit tests verifying geometric finger calculations, scale invariance, stabilization timing, anti-repeat logic, state machines, and mock OS commands.

Run the test suite:
```bash
pytest -v
```

### Expected Output:
```text
============================= test session starts ==============================
collected 21 items

tests/test_command_manager.py::test_gesture_hold_duration_requirement PASSED
tests/test_command_manager.py::test_interrupted_gesture_resets_hold_timer PASSED
tests/test_command_manager.py::test_command_cooldown_enforcement PASSED
tests/test_command_manager.py::test_prevention_of_repeated_commands_when_continuously_held PASSED
tests/test_command_manager.py::test_pause_and_resume_state_transitions PASSED
tests/test_command_manager.py::test_safe_mode_toggle_and_state PASSED
tests/test_command_manager.py::test_presentation_controller_with_mock PASSED
tests/test_command_manager.py::test_presentation_controller_safe_mode_does_not_call_os PASSED
tests/test_gesture_recognizer.py::test_recognizer_no_hand PASSED
tests/test_gesture_recognizer.py::test_recognizer_next_slide PASSED
tests/test_gesture_recognizer.py::test_recognizer_prev_slide PASSED
tests/test_gesture_recognizer.py::test_recognizer_resume_and_pause PASSED
tests/test_gesture_utils.py::test_calculate_distance PASSED
tests/test_gesture_utils.py::test_individual_finger_detection PASSED
tests/test_gesture_utils.py::test_finger_states_dictionary PASSED
tests/test_gesture_utils.py::test_gesture_next_slide PASSED
tests/test_gesture_utils.py::test_gesture_prev_slide PASSED
tests/test_gesture_utils.py::test_gesture_resume_open_palm PASSED
tests/test_gesture_utils.py::test_gesture_pause_closed_fist PASSED
tests/test_gesture_utils.py::test_unknown_gestures_handling PASSED
tests/test_gesture_utils.py::test_scale_invariance PASSED

============================== 21 passed in 0.26s ==============================
```

---

## 🛠️ Troubleshooting & FAQs

### Q1: `OpenCV: AVFoundation didn't find any attached Video Input Devices!`
- **Cause**: The computer does not detect a camera or Terminal does not have camera permissions.
- **Fix**: Open **System Settings > Privacy & Security > Camera** and check the box for Terminal / VS Code. If using an external camera, pass `--camera 1`.

### Q2: Gestures are detected on-screen, but slides don't move.
- **Cause 1**: The presentation window is not focused (the keystrokes are being sent to whatever window is active).
  - **Fix**: Click on your PowerPoint/Google Slides window.
- **Cause 2**: macOS Accessibility permissions are missing for PyAutoGUI.
  - **Fix**: Open **System Settings > Privacy & Security > Accessibility** and enable your terminal.

### Q3: How do I change the hold duration or cooldown?
- Edit `config.py`:
  ```python
  GESTURE_HOLD_SECONDS = 0.4      # Faster trigger
  COMMAND_COOLDOWN_SECONDS = 0.8  # Shorter pause
  ```

---

## 🎓 College Viva & Demonstration Guide

When presenting this project to professors, external examiners, or interviewers, highlight the following technical points:

1. **Why MediaPipe over a custom CNN?**
   - Training an end-to-end CNN classifier from raw pixels requires massive labelled datasets and is sensitive to background clutter and lighting.
   - MediaPipe decomposes the problem into a two-stage pipeline: a lightweight detector identifies the palm region, and a 3D regression network predicts 21 landmark coordinates. Our geometric module then analyzes finger bone kinematics. This achieves high accuracy with near-zero CPU overhead.
2. **How is Scale Invariance Handled?**
   - Rather than comparing absolute pixel coordinates (e.g. `tip.y < 200`), the algorithm measures relative distances compared to the hand's own anatomical base (`wrist` to `MCP` palm distance). Thus, it functions whether you are 1 meter or 3 meters away.
3. **Addressing the "Flicker" & False Trigger Problem:**
   - Explain the 4-tier anti-spam architecture (`GESTURE_HOLD_SECONDS`, `COMMAND_COOLDOWN_SECONDS`, and `REQUIRE_NEUTRAL_RELEASE`). Demonstrate on camera that holding your hand up does NOT skip 50 slides in a second.
4. **Live Demo Script:**
   - *Step 1*: Launch `python main.py --safe-mode` to show the HUD and tracking responsiveness.
   - *Step 2*: Open a sample PowerPoint or PDF slide deck on one half of the screen.
   - *Step 3*: Switch to active mode (press `S`), focus the slides, and demonstrate `Index -> Next`, `Two Fingers -> Prev`, `Fist -> Pause`, and `Palm -> Resume`.

---

## 🔮 Known Limitations & Future Improvements

- **Extreme Low-Light Conditions**: In dark rooms, webcam noise can degrade landmark detection accuracy. *Future work: IR camera support or adaptive histogram equalization (CLAHE).*
- **Multiple People in View**: If someone steps behind the presenter, background hands might compete. *Future work: Track bounding box consistency and lock onto primary presenter's face/chest region.*
- **Custom Gesture Extensions**: Swiping velocity estimation (optical flow), virtual laser pointer with fingertip tracking, and slide drawing mode.

