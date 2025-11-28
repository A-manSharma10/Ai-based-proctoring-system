# Full AI Proctoring System

A comprehensive AI-based proctoring system for online exams with advanced violation detection and real-time monitoring.

## 🚀 Features

### Core Detection Capabilities

1. **Face Detection**
   - ✅ No face detection (student leaves camera)
   - ✅ Multiple face detection (unauthorized person)
   - ✅ Profile face detection (looking sideways)

2. **Gaze & Eye Tracking**
   - ✅ Looking away from screen
   - ✅ Eyes closed detection (sleeping)
   - ✅ Real-time eye position tracking

3. **Distance Monitoring**
   - ✅ Too close to camera
   - ✅ Too far from camera
   - ✅ Calibrated baseline measurements

4. **Head Pose Estimation**
   - ✅ Head turned away detection
   - ✅ Face orientation analysis
   - ✅ Aspect ratio monitoring

5. **Advanced Features**
   - ✅ Real-time violation alerts
   - ✅ Audio alerts (Windows beep)
   - ✅ Screenshot capture on violations
   - ✅ Comprehensive session reports
   - ✅ Violation severity levels (CRITICAL, HIGH, MEDIUM)
   - ✅ Visual UI with live statistics

## 📋 Requirements

```
Python 3.x
OpenCV (opencv-python)
NumPy
```

Already installed in your environment!

## 🎯 Usage

### Run the Full System

```bash
python full_proctoring_system.py
```

### During the Session

1. **Calibration Phase** (3 seconds)
   - Look directly at the screen
   - Sit at normal distance (2-3 feet)
   - Ensure good lighting

2. **Monitoring Phase**
   - System actively monitors all violations
   - Real-time alerts appear on screen
   - Audio beeps for critical violations

3. **End Session**
   - Press **ESC** to end
   - Report automatically generated

## 🔍 Detected Violations

| Violation Type | Severity | Description |
|---------------|----------|-------------|
| **NO_FACE** | HIGH | Student leaves camera view |
| **MULTIPLE_FACES** | CRITICAL | More than one person detected |
| **PROFILE_DETECTED** | MEDIUM | Face turned sideways |
| **LOOKING_AWAY** | MEDIUM | Eyes not on screen |
| **EYES_CLOSED** | HIGH | Eyes closed for extended period |
| **HEAD_TURNED** | MEDIUM | Head orientation suspicious |
| **TOO_CLOSE** | MEDIUM | Too close to camera |
| **TOO_FAR** | MEDIUM | Too far from camera |

## 📊 Output Files

### Reports
- `reports/session_YYYYMMDD_HHMMSS.txt` - Detailed session report with:
  - Session duration
  - Total violations
  - Breakdown by type and severity
  - Detailed violation log with timestamps
  - Overall assessment

### Violation Screenshots
- `violations/` - Screenshots of all violations
- Format: `sessionID_VIOLATIONTYPE_timestamp.jpg`

## ⚙️ Customization

Edit threshold values in the code:

```python
# In FullProctoringSystem.__init__()
self.NO_FACE_THRESHOLD = 30          # frames (~1 second)
self.MULTIPLE_FACE_THRESHOLD = 15    # frames (~0.5 seconds)
self.LOOKING_AWAY_THRESHOLD = 45     # frames (~1.5 seconds)
self.EYES_CLOSED_THRESHOLD = 60      # frames (~2 seconds)
self.HEAD_TURNED_THRESHOLD = 30      # frames (~1 second)
```

## 💡 Tips for Best Results

1. **Lighting**
   - Ensure face is well-lit
   - Avoid backlighting
   - Use natural or white light

2. **Camera Position**
   - Place camera at eye level
   - Sit 2-3 feet away
   - Center yourself in frame

3. **Environment**
   - Clear background
   - Minimize movement
   - Quiet space

4. **During Exam**
   - Look at screen naturally
   - Avoid sudden movements
   - Don't leave camera view

## 🎨 UI Features

### On-Screen Display
- **Top Bar**: System status and monitoring info
- **Top Right**: Violation counter and timer
- **Left Panel**: Real-time violation summary
- **Bottom Banner**: Active violation warnings (color-coded by severity)

### Color Coding
- 🟢 Green: Normal operation
- 🟡 Yellow: Medium severity violation
- 🟠 Orange: High severity violation
- 🔴 Red: Critical severity violation

## 📈 Assessment Levels

The system provides an overall assessment:

- **EXCELLENT**: 0 violations
- **GOOD**: 1-5 violations
- **FAIR**: 6-15 violations
- **POOR**: 16-30 violations
- **CRITICAL**: 30+ violations

## 🔧 Advanced Features (Optional)

### For Object Detection (Phones, Books, etc.)
To enable object detection, download YOLO:

1. Download YOLOv3-tiny:
   - `yolov3-tiny.weights`
   - `yolov3-tiny.cfg`
   - `coco.names`

2. Place in project directory

3. Run `complete_proctoring_system.py` instead

### For Audio Monitoring
Install additional libraries:

```bash
pip install pyaudio SpeechRecognition
```

Then use `complete_proctoring_system.py` for full audio monitoring.

## 🛡️ Privacy & Security

- ✅ All processing done locally
- ✅ No data sent to external servers
- ✅ Recordings stay on your machine
- ✅ You control all data

## 📝 System Requirements

- **OS**: Windows (audio alerts), Linux, macOS
- **Python**: 3.7+
- **Webcam**: Required
- **RAM**: 2GB minimum
- **CPU**: Dual-core minimum

## 🐛 Troubleshooting

### Camera not detected
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Poor detection accuracy
- Improve lighting
- Clean camera lens
- Adjust distance from camera
- Remove glasses if possible

### High false positives
- Increase threshold values
- Ensure stable seating position
- Minimize background movement

## 📄 License

MIT License - Feel free to modify and use for your needs.

## 🤝 Contributing

Suggestions and improvements welcome!

---

**Note**: This system is designed for educational purposes and should be used ethically with proper consent from all participants.
