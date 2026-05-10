# AI-Based Online Exam Proctoring System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview
A comprehensive AI-powered online exam proctoring platform that uses computer vision, audio analysis, and behavioral monitoring to ensure exam integrity while maintaining student privacy. The system seamlessly integrates real-time monitoring, an automated violation detection pipeline, and supervisor controls. 

## System Architecture

The architecture relies on a microservices design with real-time WebSocket communication:
- **Frontend**: A React.js application offering dedicated portals for Students, Supervisors, and Admins.
- **Backend API Gateway**: A Node.js/Express server that orchestrates data flow, manages exam states, and broadcasts live events via Socket.IO.
- **AI Microservices**: Python (FastAPI) services handling distinct detection modules (Face, Audio, Object, Behavior).
- **Database**: MySQL and Redis (or MockDB for local dev) for secure exam session storage, user profiles, and logs.

## Technology Stack

### Frontend
- **React.js 18** (with React Router)
- **TailwindCSS** for responsive UI
- **Socket.IO-client** for real-time live monitoring
- **WebRTC** for video streaming

### Backend
- **Node.js** & **Express.js**
- **Socket.IO** for event-driven WebSockets
- **JWT Authentication**

### AI Services
- **Python 3.9+** & **FastAPI**
- **PyTorch / TensorFlow**
- **OpenCV & MediaPipe**
- **YOLOv8** for object detection

### Database & Deployment
- **MySQL 8.0**
- **Docker & Docker Compose**

## Features

- **Real-Time Monitoring System**: Continuous gaze tracking, face presence, and behavioral analysis.
- **LiveMonitoring.js Dashboard**: A specialized supervisor view displaying multiple concurrent student feeds.
- **Socket-Based Video Streaming**: Low-latency video frame transmission for instantaneous AI inferences.
- **Violation Detection Pipeline**: Detects anomalies like missing face, multiple people, looking away, unauthorized devices, and audio alerts.
- **Screenshot Evidence Capture**: Automatically records keyframes at the exact moment of a violation.
- **Experiment Comparison Framework**: Compares the effectiveness of Single-Modal vs. Multimodal configurations.
- **Admin Control Panel**: Interface to configure test blueprints and review system analytics.
- **Supervisor Dashboard**: Active session management with the ability to pause or terminate student exams.
- **Exam Termination Logic**: Strict rules that auto-terminate exams upon reaching the maximum defined threshold of warnings.
- **Integrity Scoring**: Generates an automated trust score summarizing user behavior during the session.

## Getting Started

### 1. Prerequisites
- **Docker and Docker Compose** (Highly Recommended)
- Node.js (v18+) and Python (v3.9+) (Only required for manual setup)

### 2. Clone the repository
```bash
git clone https://github.com/A-manSharma10/Ai-based-proctoring-system
cd Ai-based-proctoring-system
```

## How to Run the System

### Method 1: Using Docker (Recommended)
The simplest way to run the entire stack (Frontend, Backend, AI Microservices, and Databases) is using Docker Compose:

```bash
docker-compose up --build
```
Once the containers are running, the application will be accessible at:
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:5000`

### Method 2: Manual Local Startup
If you prefer to run the services without Docker, you will need to install the dependencies and start manually:

1. **Install Dependencies:**
   - Backend: `cd backend && npm install`
   - Frontend: `cd frontend && npm install`
   - AI Services: `cd ai_services && pip install -r requirements.txt`

2. **Start the Backend:**
   ```bash
   cd backend
   npm run dev
   ```

3. **Start the Frontend:**
   ```bash
   cd frontend
   npm start
   ```
*(Note: For the manual setup to work fully, ensure your local MySQL/Redis databases and the individual Python AI microservices are running on their respective ports.)*

## How to Run Experiments

The repository includes a dedicated `experiments/` testing framework to validate AI model effectiveness. 

```bash
# Run Single-Modal tests (Vision Only)
python -m experiments.experiment_runner --mode single_modal

# Run Multimodal tests (Vision + Audio + Behavior)
python -m experiments.experiment_runner --mode multimodal
```
The experiment outputs calculate Precision, Recall, Accuracy, and F1-Scores.

## Research Comparison (Single Modal vs Multimodal)

Our research demonstrates significant improvements when replacing a standard Single-Modal proctoring setup with a Multimodal approach:

- **Single-Modal Proctoring**: Relies strictly on face detection and object tracking. Often generates false positives when a user legitimately looks away to solve a problem on paper.
- **Multimodal Proctoring**: Integrates spatial gaze tracking, ambient audio context, and behavioral posture. 

**Conclusion**: The Multimodal pipeline dramatically lowers the **False Alert Rate** while preserving high accuracy, as it cross-references visual anomalies with auditory cues prior to raising a violation flag.

## Demo Instructions

To test the system functionality locally:
1. Start the application using Docker (`docker-compose up`).
2. Open two browser windows:
   - **Student Login**: `student1@exam.com` / `password`
   - **Supervisor Login**: `supervisor@exam.com` / `password`
3. Have the student "Start" an exam. The examiner can then go to **Live Monitoring** to watch the feed.
4. Intentionally look away from the camera or introduce a phone to witness the real-time alerting mechanics.

## Screenshots

*Example snapshots showcasing the system capabilities.*
- **Accuracy Comparison**: `experiments/results/accuracy_comparison.png`
- **False Alert Rate Chart**: `experiments/results/false_alert_rate.png`

## Future Improvements

- **Cloud Deployment Optimization**: Migrate AI inference models directly to AWS SageMaker/Azure ML for better horizontal scaling.
- **Advanced Action Recognition**: Enhance the behavioral AI block with skeleton tracking to identify highly specific cheating actions.
- **Offline Integrity**: Implement local caching of events for students experiencing brief network disconnects, syncing securely upon reconnection.