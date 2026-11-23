# 🎯 Object Detection and Tracking System

An AI-powered Object Detection and Tracking System built using YOLOv8 and ByteTrack. The application detects objects, tracks them across video frames, counts people, and provides real-time detection statistics through an interactive Streamlit interface.

## 📌 Project Overview

This project uses computer vision and object tracking techniques to detect and monitor objects in videos and live webcam streams.

It provides two input modes:
- Upload Video: Upload a video file for object detection and tracking.
- Live Webcam: Perform real-time object detection and tracking using a webcam.

## ✨ Features

- **Object Detection:** Detect multiple objects using YOLOv8.
- **Object Tracking:** Track detected objects across video frames using ByteTrack.
- **People Counting:** Count unique detected people.
- **Live Statistics:** Display currently tracked objects and processed frames.
- **Crowd Monitoring:** Monitor crowd levels using the webcam.
- **Crowd Capacity:** Set a maximum crowd capacity for webcam monitoring.
- **Crowd Status:** Classify crowd levels as Low, Moderate, High, or Overcrowded.
- **Detection History:** Maintain a record of detected objects.
- **Downloadable Reports:** Download detection reports.

## 🛠️ Technologies Used

- Python
- YOLOv8
- ByteTrack
- OpenCV
- Streamlit
- Streamlit-WebRTC
- NumPy
- Pandas

## 📂 Project Structure

```text
ObjectDetectionTracking/
│
├── app.py
├── detector.py
├── tracker.py
├── custom_bytetrack.yaml
├── requirements.txt
├── yolov8n.pt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ipshita3106/object-detection-tracking.git
```

### 2. Open the Project Folder

```bash
cd object-detection-tracking
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Detection and Monitoring

The system provides:
- Total unique detected objects
- Currently tracked objects
- Total processed frames
- People counting
- Webcam crowd monitoring and crowd status

## 🎯 Project Objective

The objective of this project is to develop a real-time computer vision application that combines object detection, object tracking, people counting, and crowd monitoring in a user-friendly interface.

## 👩‍💻 Developer

**Ipshita**

B.Tech Computer Science and Engineering (Artificial Intelligence)

GitHub: https://github.com/ipshita3106

---

⭐ If you find this project interesting, consider giving it a star!