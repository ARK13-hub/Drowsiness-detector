# Driver Drowsiness Detection System

A real-time computer vision system that monitors driver attentiveness and drowsiness using eye aspect ratio (EAR) and mouth aspect ratio (MAR) calculations with facial landmark prediction. When signs of drowsiness or prolonged eye closure / yawning are detected, an audible alarm triggers.

## Features
- Real-time webcam video stream processing.
- Facial landmark detection using dlib (68 landmark points).
- Eye Aspect Ratio (EAR) monitoring for eye closure/blinking frequency.
- Mouth Aspect Ratio (MAR) monitoring for yawning detection.
- Audio alarm playback using `pygame.mixer`.
- Rotating activity and event logging.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ARK13-hub/Drowsiness-detector.git
   cd Drowsiness-detector
   ```

2. **Create and activate a virtual environment (Python 3.10 or 3.11 recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained landmark predictor model:**
   Run the included helper script:
   ```bash
   python download_model.py
   ```
   *(Or manually download `shape_predictor_68_face_landmarks.dat` from [dlib-models](https://github.com/davisking/dlib-models) and place it in the project root).*

---

## Running the Application

Ensure your webcam is connected, then start the detector:
```bash
python main.py
```

- Press `ESC` in the video window to quit.
