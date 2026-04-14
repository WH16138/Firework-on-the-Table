# 🎆 Fireworks on My Table

Real-time Augmented Reality fireworks rendered on a physical chessboard using webcam-based pose estimation.

---

## Overview

**Fireworks on My Table** is a real-time computer vision and augmented reality project that detects a chessboard through a webcam, estimates the camera pose relative to the board, and overlays dynamic AR fireworks directly onto the detected surface.

This project implements a complete AR pipeline using only **Python** and **OpenCV**, including camera calibration, pose estimation, and a custom particle-based firework renderer.

---

## Features

* Automatic chessboard image collection for calibration
* Camera intrinsic calibration with distortion correction
* Real-time chessboard detection and pose estimation
* Multi-firework particle simulation
* Glow / Trail / Afterimage rendering effects
* Stable pose tracking with temporary loss tolerance
* Optimized batch projection for real-time performance

---

## Demo Video

> Add your demo video link below

```text
https://your-demo-video-link-here.com
```

---

## Project Structure

```text
.
├── capture_chessboard.py        # Automatically capture valid calibration images
├── calibrate_camera.py          # Camera calibration from captured chessboard images
├── pose_ar_fireworks.py         # Main AR fireworks application
│
├── calibration_data.npz         # Saved calibration result
│
├── raw_chessboards/             # Raw captured chessboard frames
├── best_chessboards/        # Filtered images used for calibration
│
└── README.md
```

---

## File Descriptions

### `capture_chessboard.py`

Automatically captures webcam frames containing a valid chessboard pattern.

**Purpose**

* Collect calibration images efficiently
* Save only frames with detectable chessboard patterns

---

### `calibrate_camera.py`

Performs intrinsic camera calibration using the selected chessboard images.

**Outputs**

* Camera Matrix
* Distortion Coefficients
* RMS Reprojection Error
* Saved calibration file (`calibration_data.npz`)

---

### `pose_ar_fireworks.py`

Main augmented reality application.

**Responsibilities**

* Detect chessboard in real time
* Estimate camera pose with `solvePnP`
* Simulate multi-firework particle system
* Render AR fireworks aligned to the chessboard plane

---

## How to Run

### 1. Capture Calibration Images

```bash
python capture_chessboard.py
```

Move the chessboard to various positions/angles in front of the webcam.

Captured images will be saved to:

```text
raw_chessboards/
```

---

### 2. Calibrate Camera

```bash
python calibrate_camera.py
```

Calibration result will be saved as:

```text
calibration_data.npz
```

---

### 3. Run AR Fireworks Demo

```bash
python pose_ar_fireworks.py
```

Show the chessboard to the webcam to begin AR rendering.

---

## Technical Pipeline

### 1. Camera Calibration

Uses multiple chessboard images to estimate:

* Intrinsic Camera Matrix
* Lens Distortion Parameters

via OpenCV calibration routines.

---

### 2. Pose Estimation

Real-time pose is estimated through:

* `cv.findChessboardCorners`
* `cv.cornerSubPix`
* `cv.solvePnP`

producing 6DoF camera pose relative to the chessboard.

---

### 3. AR Firework Simulation

Custom particle system featuring:

* Independent firework rockets
* Randomized launch trajectories
* Explosion particle emission
* Per-firework coherent color palette
* Glow and trail effects

---

## Dependencies

```bash
pip install opencv-python numpy
```

---

## Requirements

* Python 3.10+
* Webcam
* Printed / Physical Chessboard Pattern

---

## Future Improvements

* Sprite-based glow rendering for better performance
* More firework burst patterns
* AR object occlusion handling
* Markerless plane tracking
* Sound effect integration

---

## Author

Created by **[Ju Hyeon Seong]**

---

## License

MIT License
