# Project Guide: AI-Based Online Exam Proctoring System

This extensive guide explains the core features, logical flow, and development workflows for the AI-Based Online Exam Proctoring System.

## 1. Folder Structure

The repository has been structured for scalability and microservice separation:

```
/Proctoring system
├── /ai_services            # Python AI processing nodes
│   ├── /audio_analysis     # Speech / multiple voices detection
│   ├── /behavioral_analysis# Gaze, posture anomaly tracking
│   ├── /face_recognition   # Presence verification (ArcFace/MediaPipe)
│   └── /object_detection   # Prohibited item scanning (YOLO)
├── /backend                # Central Node.js server orchestrating clients
│   ├── /config             # Auth and environment management
│   ├── /middleware         # JWT protection logic
│   ├── /routes             # RESTful API endpoints for exams, users, reports
│   ├── /socket             # WebSocket broadcasting definitions
│   └── server.js           # Server entry point
├── /database               # Core database schemas and migrations
├── /docs                   # Technical specs and architecture plans
├── /experiments            # Benchmark for Single-Modal vs Multimodal
│   ├── /dataset            # Video/audio samples for evaluating the model
│   ├── /results            # Exported JSON metrics and charts
│   └── experiment_runner.py# Pipeline script to test model efficacy
├── /frontend               # React interface for all system roles
│   ├── /public             # Static assets
│   └── /src
│       ├── /components     # UI building blocks (Modals, Headers, Forms)
│       ├── /contexts       # Global state (Auth/Role management)
│       └── /pages          # Main app views (LiveMonitoring, StudentDashboard)
├── run_proctoring_system.bat   # Startup batch script (development use)
├── PROJECT_GUIDE.md        # This guide
└── README.md               # Quick start and configuration details
```

## 2. System Architecture

A modular distributed system design minimizes inference latency:
1. **Frontend Presentation**: Students perform exams on modern web browsers sharing a live camera stream. Supervisor dashboards receive events/states.
2. **WebSocket Highway**: Built atop WebRTC (for peer video feeds when required) and Socket.IO to manage states efficiently without long-polling. 
3. **Core Backend Logic**: Validates interactions, processes auth, and triggers AI analysis services directly or queues up async evaluation tools.
4. **AI Microservices Layer**: Operates independently (via FastAPI), providing isolated execution environments for heavy machine learning model inference (yolov8, face tracking).

## 3. Frontend Workflow

The React 18 SPA (Single Page Application) handles state with Context API and visual layout natively with TailwindCSS:
- **Student Flow**: `/login` (Auth) -> `/student-dashboard` (View tasks) -> `/exam/:id` (Real-time protected environment with automated camera verification and local event dispatchers).
- **Supervisor Flow**: `/login` -> `/supervisor/live-monitoring` (Real-time multi-feed observing) -> `/supervisor/alerts` (Immediate automated warnings).
- **Admin Flow**: `/login` -> `/admin-dashboard` (Manage courses, review historic audits at the `/report-center`).

## 4. Backend APIs

The backend leverages an Express.js router framework alongside strong typing via Socket abstractions:
- **`GET /api/exams`**: Pulls a list of available/archived assessments globally.
- **`POST /api/auth/login`**: Authenticates users and dispenses a signed JWT.
- **`POST /api/alerts/report`**: Internally records an AI-detected rule violation against a specific student session ID.
- **`GET /api/reports/user/:id`**: Resolves historical metrics alongside generated screenshot URI paths.

## 5. AI Detection Modules

Separated logically to compute isolated scores:
- **Face & Gaze Tracker**: Scans frame-by-frame ensuring face is centered on the exam screen.
- **Audio Interpreter**: Scans background noise floor for whispered conversations or secondary speakers.
- **Object Scanner**: Triggers immediately on rendering boundaries denoting "Phone", "Book", or "Tablet" overlapping the student's background space.
- **Behavioral Rater**: Merges multiple risk indexes over a timeline, producing a final "trust score."

## 6. Experiment Framework

Our evaluation platform located inside `/experiments` runs offline benchmarking:
- Includes raw `dataset_loader.py` to iterate through known cheating and non-cheating multimedia samples.
- Executes `experiment_runner.py`, triggering single_modal and multimodal scenarios to measure false positives and true negatives.
- Exports metrics natively as images (e.g. `accuracy_comparison.png`) to `/experiments/results` via `report_generator.py`.

## 7. Testing Instructions

To ensure system reliability during deployment:
- Execute `npm test` inside both backend and frontend environments.
- Run the python integration tests natively in the AI layers: `python -m pytest /ai_services`.
- Healthcheck scripts (`health-check.js`) systematically ping internal microservice boundaries. 

## 8. Demo Instructions

The fastest way to experience the features locally:

### Run the App
```bash
./run_proctoring_system.bat
```

### Emulate the Environment
1. Log into Chrome via User A: `student1@exam.com` / `password`. Open a real exam instance. Look away from the camera for an extended duration.
2. Log into a private window or Edge via User B: `supervisor@exam.com` / `password`. Observe the dashboard under the **Live Monitoring** panel.
3. Once the student garners enough warnings, watch the system automatically flag the stream red and dispatch an immediate `EXAM TERMINATED` global state.
