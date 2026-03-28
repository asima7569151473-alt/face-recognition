# Real-Time FaceGuard AI Pro

FaceGuard AI Pro is a real-time, premium computer vision application built with Streamlit, OpenCV, DeepFace, and MediaPipe. It provides a state-of-the-art neural surveillance suite supporting biometric facial identification, kinetic hand gesture tracking, and anti-spoofing security.

## Features
- **Face Recognition Live:** Real-time facial identification with active blink/liveness detection.
- **Hand Gesture Live:** Kinetic skeletal tracking of hands for potential UI abstraction.
- **Biometric Enrollment:** Register new identities into a secure localized SQLite embeddings database via webcam or image upload.
- **Group Photo Match:** Scan high-density environment images using YOLOv8 to locate targeted facial signatures rapidly.
- **Premium UI:** A stunning, modern dark-themed dashboard featuring multi-layered glassmorphism panels.

## Tech Stack
- **Python 3.10+**
- **Streamlit** (Web App Framework & UI)
- **OpenCV** (Video Feed & Frame Manipulation)
- **DeepFace** (Facial Embeddings & Recognition using VGG-Face/Facenet)
- **MediaPipe Tasks API** (Face Mesh for Anti-Spoofing & Hand Landmarks)
- **Ultralytics YOLOv8** (Person Detection)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/asima7569151473-alt/face-recognition.git
   cd face-recognition
   ```

2. **Install system dependencies:**
   Ensure you have Python installed. If you are on Windows, you may need the C++ Build Tools installed for certain OpenCV or deep learning dependencies.

3. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: MediaPipe requires `mediapipe.tasks.python` which is supported on Python 3.9 - 3.12).*

4. **Initialize the local database:**
   The SQLite database (`face_db.sqlite`) will automatically be initialized the first time you run the application to store user biometric embeddings.

## Run Command
Execute the following command in your terminal to start the Streamlit server:
```bash
streamlit run app.py
```
The application will launch on `http://localhost:8501`.
