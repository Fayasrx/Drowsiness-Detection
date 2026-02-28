# Drowsiness Detection System

A real-time drowsiness detection system built with Python that monitors a driver's eyes through a webcam and triggers an alarm when signs of drowsiness are detected. This project implements multiple detection approaches using both Haar Cascades and Dlib facial landmarks.

## Features

- **Real-time face and eye detection** via webcam feed
- **Eye Aspect Ratio (EAR)** calculation for accurate blink/drowsiness detection
- **Audio alarm** triggered when drowsiness is detected
- **Multiple detection approaches** included for comparison:
  - Haar Cascade-based detection (`DrowsinessDetection.py`)
  - Dlib 68/70-point facial landmark detection (`blinkDetect.py`, `sur.py`, `sur2.py`, `sur4.py`, `surya3.py`)
- **Blink counting** and drowsiness duration tracking
- **Gamma correction & histogram equalization** for improved detection under varying lighting

## How It Works

1. The webcam captures video frames in real time.
2. Faces are detected in each frame using either Haar Cascades or Dlib's frontal face detector.
3. Facial landmarks are extracted to locate the eye regions.
4. The **Eye Aspect Ratio (EAR)** is computed — when eyes are closed, EAR drops below a threshold.
5. If the EAR stays below the threshold for a sustained number of frames, a **drowsiness alert** is triggered with an audible alarm.

## Project Structure

```
├── DrowsinessDetection.py   # Haar Cascade-based drowsiness detection
├── blinkDetect.py           # Dlib-based blink & drowsiness detection (70 landmarks)
├── sur.py                   # Dlib-based EAR detection (68 landmarks)
├── sur2.py                  # Enhanced version with additional features
├── sur4.py                  # Variant with refined thresholds
├── surya3.py                # Alternative implementation
├── bhsdbj.py                # Supporting detection script
├── alarm.wav                # Alarm sound file (WAV)
├── alarm_sound.mp3          # Alarm sound file (MP3)
├── .gitignore
└── README.md
```

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- Dlib
- NumPy
- SciPy
- imutils
- pygame
- playsound

### Install Dependencies

```bash
pip install opencv-python dlib numpy scipy imutils pygame playsound
```

> **Note:** Dlib may require CMake and a C++ compiler to install. On Windows, you may need to install Visual Studio Build Tools first.

## Usage

### Quick Start (Haar Cascade approach — no extra models needed)

```bash
python DrowsinessDetection.py
```

### Dlib-based approach (more accurate, requires landmark model)

1. Download the shape predictor model:
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place it in the project directory or a `models/` folder.

2. Run:
```bash
python blinkDetect.py
```

### Controls

- Press **`q`** to quit the application.

## Detection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EYE_AR_THRESH` | 0.25 - 0.27 | EAR threshold below which eyes are considered closed |
| `FRAMES_THRESHOLD` | 15 - 48 | Consecutive closed-eye frames before alarm triggers |
| `drowsyTime` | 1.5s | Duration of closed eyes to confirm drowsiness |
| `blinkTime` | 0.15s | Minimum blink duration |

## Technologies Used

- **OpenCV** — Video capture and image processing
- **Dlib** — Facial landmark detection
- **SciPy** — Euclidean distance for EAR calculation
- **Pygame / Playsound** — Audio alarm playback

## License

This project is open source and available for educational and research purposes.
