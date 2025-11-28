# AI Proctoring System

A comprehensive AI-based proctoring system for online exams that detects suspicious behavior in real-time.

## Features

### Basic Proctoring System (`proctoring_system.py`)
- ✅ Real-time face detection
- ✅ Eye gaze tracking
- ✅ Looking away detection
- ✅ Multiple face detection
- ✅ Warning system with logs
- ✅ Session logging to file

### Advanced Proctoring System (`advanced_proctoring.py`)
- ✅ All basic features
- ✅ Audio alerts on violations
- ✅ Screenshot capture on violations
- ✅ Eyes closed/sleeping detection
- ✅ Detailed session reports
- ✅ Visual progress indicators
- ✅ Severity-based violation tracking

## Installation

Already installed! Just run the scripts.

## Usage

### Basic System
```bash
python proctoring_system.py
```

### Advanced System (Recommended)
```bash
python advanced_proctoring.py
```

Press **ESC** to end the session.

## Detected Violations

1. **Looking Away** - When eyes deviate significantly from screen center
2. **No Face Detected** - Student leaves the camera frame
3. **Multiple Faces** - More than one person detected
4. **Eyes Closed** - Prolonged eye closure (sleeping detection)

## Output Files

### Logs
- `proctoring_log.txt` - Real-time event log (basic system)

### Reports
- `reports/session_YYYYMMDD_HHMMSS.txt` - Detailed session report

### Violations
- `violations/` - Screenshots of all violations with timestamps

## Thresholds (Adjustable)

- Looking Away: 45 frames (~1.5 seconds)
- No Face: 30 frames (~1 second)
- Multiple Faces: 15 frames (~0.5 seconds)
- Eyes Closed: 60 frames (~2 seconds)

## Tips for Best Results

1. Ensure good lighting on your face
2. Position camera at eye level
3. Sit 2-3 feet from camera
4. Avoid wearing glasses if possible (reduces accuracy)
5. Keep background clear

## Customization

Edit threshold values in the code:
```python
self.LOOKING_AWAY_LIMIT = 45  # Adjust sensitivity
self.NO_FACE_LIMIT = 30
self.MULTIPLE_FACE_LIMIT = 15
```

## System Requirements

- Python 3.x
- OpenCV (already installed)
- Webcam
- Windows OS (for audio alerts)

## Notes

- The system runs entirely offline
- No data is sent to external servers
- All recordings stay on your local machine
- Violations are logged with timestamps for review
