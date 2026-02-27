# FaceVault
**AI-Secured Premium Entry System**

## Short Description
FaceVault is a real-time, AI-powered face recognition system built for premium lounges and secure environments. It allows administrators to register members using a simple photo upload and instantly verifies their identity using a live camera feed. 

## Problem Statement
Traditional entry methods like ID cards, PINs, or physical tickets are easily lost, shared, or stolen. They also slow down entry at premium venues where speed and a seamless user experience are critical. 

## Solution Overview
FaceVault eliminates the need for physical IDs by using advanced facial recognition. Members simply walk up to the camera, and the system instantly grants or denies access based on a secure, local database of facial embeddings. The system is fast, secure, and provides a premium "hands-free" entry experience.

## Features
* **Real-Time Verification:** Instant face detection and matching at over 30 FPS.
* **Premium Dashboard:** A beautiful, animated WebGL dashboard to manage members and monitor live scans.
* **Easy Registration:** Add a member instantly by uploading a single JPG or PNG photo.
* **Deep Security:** Uses ArcFace AI and ResNet-10 SSD to accurately map facial features.
* **Distance Detection:** Automatically rejects scans if a person is standing too far away.
* **Live Entry Logs:** Keeps an automatic record of everyone who gets granted or denied access. 

## Tech Stack
* **Frontend:** HTML5, Vanilla JavaScript, CSS3, WebGL (Aurora Shader)
* **Backend:** Python, Flask
* **AI & Computer Vision:** OpenCV, DeepFace (ArcFace model, Retinaface detector)
* **Database:** SQLite (for member metadata), Pickle (for fast vector embeddings)

## How It Works
1. **Registration:** You upload a standard photo of a member's face via the dashboard.
2. **Feature Extraction:** FaceVault extracts the unique facial features (embeddings) and saves them securely.
3. **Live Scanning:** A webcam constantly monitors the environment using a fast tracking algorithm.
4. **Matching:** When a face is detected close enough, the system compares it against the secure database.
5. **Decision:** Access is either GRANTED (with a green highlight and member info) or DENIED (with a red highlight).

## Installation & Setup Instructions

**Prerequisites:** Python 3.9+ and a working webcam.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/facevault.git
   cd facevault
   ```

2. **Install the required Python packages:**
   ```bash
   pip install flask flask-cors opencv-python deepface numpy
   ```

3. **Download the model files (if missing):**
   * Ensure `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` are in the main project folder.

4. **Start the server:**
   ```bash
   python app.py
   ```

## Usage Instructions
1. Open your browser and go to `http://localhost:5000`.
2. Go to the **Register Member** tab to add a new person by uploading their photo and details.
3. Go to the **Live Verify** tab and click **Start Camera**.
4. Stand in front of the camera to see the system recognize you instantly!
5. Use the Trash icon in the Entry Log header to clear old scan history.

## Future Improvements
* Add Liveness Detection (anti-spoofing) to prevent people from holding up photos to the camera.
* Deploy the dashboard to the cloud while keeping camera processing on local edge devices.
* Add email or SMS notifications when VIP members are detected.
* Support multiple camera feeds simultaneously.

## Contributors
* [Team Dominators Pro Max]
