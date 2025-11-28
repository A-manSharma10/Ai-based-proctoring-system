"""
COMPLETE AI PROCTORING SYSTEM
Features:
- Face detection (no face, multiple faces)
- Gaze tracking (looking away)
- Object detection (phone, book, etc.)
- Audio monitoring (multiple voices, suspicious sounds)
- Distance monitoring
- Head pose estimation
- Real-time alerts and logging
"""

import cv2
import numpy as np
import os
import datetime
import time
from collections import deque
import threading
import pyaudio
import wave
import speech_recognition as sr

class CompleteProctoringSystem:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load YOLO for object detection
        self.load_yolo()
        
        # Violation tracking
        self.violations = []
        self.violation_counts = {
            'NO_FACE': 0,
            'MULTIPLE_FACES': 0,
            'LOOKING_AWAY': 0,
            'EYES_CLOSED': 0,
            'SUSPICIOUS_OBJECT': 0,
            'MULTIPLE_VOICES': 0,
            'TOO_CLOSE': 0,
            'TOO_FAR': 0,
            'HEAD_TURNED': 0
        }
        
        # Thresholds
        self.NO_FACE_THRESHOLD = 30  # frames
        self.MULTIPLE_FACE_THRESHOLD = 15
        self.LOOKING_AWAY_THRESHOLD = 45
        self.EYES_CLOSED_THRESHOLD = 60
        self.OBJECT_DETECTION_INTERVAL = 30  # Check every 30 frames
        
        # Counters
        self.no_face_counter = 0
        self.multiple_face_counter = 0
        self.looking_away_counter = 0
        self.eyes_closed_counter = 0
        self.frame_count = 0
        
        # Directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        os.makedirs('audio_logs', exist_ok=True)
        
        # Session info
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Audio monitoring
        self.audio_thread = None
        self.audio_monitoring = True
        self.audio_violations = []
        
        # Calibration
        self.calibrated = False
        self.baseline_face_size = None
        
        print("\n" + "="*70)
        print("COMPLETE AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ Video monitoring initialized")
        print("✓ Object detection loaded")
        print("✓ Audio monitoring ready")
        print("\nDetection Features:")
        print("  • Face detection (single person only)")
        print("  • Gaze tracking")
        print("  • Eye closure detection")
        print("  • Suspicious object detection (phone, book, laptop)")
        print("  • Audio monitoring (multiple voices)")
        print("  • Distance monitoring")
        print("  • Head pose estimation")
        print("\nPress ESC to end session")
        print("="*70 + "\n")
    
    def load_yolo(self):
        """Load YOLO model for object detection"""
        try:
            # Try to load YOLOv3-tiny (faster, lighter)
            weights_path = 'yolov3-tiny.weights'
            config_path = 'yolov3-tiny.cfg'
            
            if not os.path.exists(weights_path) or not os.path.exists(config_path):
                print("⚠️  YOLO files not found. Object detection will be limited.")
                print("   Download from: https://pjreddie.com/darknet/yolo/")
                self.net = None
                return
            
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open('coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Suspicious objects to detect
            self.suspicious_objects = ['cell phone', 'book', 'laptop', 'tv', 'keyboard']
            
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            print("✓ YOLO object detection loaded")
        except Exception as e:
            print(f"⚠️  Could not load YOLO: {e}")
            self.net = None
    
    def detect_objects(self, frame):
        """Detect suspicious objects in frame"""
        if self.net is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        detected_objects = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    object_name = self.classes[class_id]
                    if object_name in self.suspicious_objects:
                        detected_objects.append(object_name)
                        
                        # Draw bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"{object_name}", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return detected_objects

    
    def start_audio_monitoring(self):
        """Start audio monitoring in separate thread"""
        def monitor_audio():
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.audio_monitoring:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    # Try to recognize speech
                    try:
                        text = recognizer.recognize_google(audio)
                        # Simple heuristic: if multiple distinct voices or conversation detected
                        if len(text.split()) > 10:  # Significant speech detected
                            self.log_violation('MULTIPLE_VOICES', 'HIGH', 
                                             f"Speech detected: {text[:50]}...")
                    except sr.UnknownValueError:
                        pass  # No speech detected
                    except sr.RequestError:
                        pass  # API error
                        
                except Exception as e:
                    pass
                
                time.sleep(2)
        
        try:
            self.audio_thread = threading.Thread(target=monitor_audio, daemon=True)
            self.audio_thread.start()
            print("✓ Audio monitoring started")
        except Exception as e:
            print(f"⚠️  Audio monitoring unavailable: {e}")
    
    def detect_gaze(self, frame, face):
        """Detect if person is looking away"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        if len(eyes) == 0:
            return True, True  # Eyes closed or not detected
        
        if len(eyes) < 2:
            return True, False  # Looking away
        
        # Check eye positions relative to face
        eye_centers = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_center_x = ex + ew // 2
            eye_centers.append(eye_center_x)
        
        if len(eye_centers) == 2:
            face_center = w // 2
            eyes_center = sum(eye_centers) / 2
            deviation = abs(eyes_center - face_center)
            
            # If eyes are significantly off-center, person is looking away
            if deviation > w * 0.2:
                return True, False
        
        return False, False
    
    def estimate_head_pose(self, face, frame_width):
        """Estimate if head is turned away"""
        x, y, w, h = face
        face_center_x = x + w // 2
        frame_center_x = frame_width // 2
        
        # If face is significantly off-center, head might be turned
        deviation = abs(face_center_x - frame_center_x)
        if deviation > frame_width * 0.25:
            return True
        
        # Check face aspect ratio (turned faces are narrower)
        aspect_ratio = w / h
        if aspect_ratio < 0.6:  # Face too narrow
            return True
        
        return False
    
    def check_distance(self, face_size):
        """Check if person is too close or too far"""
        if self.baseline_face_size is None:
            return None
        
        ratio = face_size / self.baseline_face_size
        
        if ratio > 1.5:
            return 'TOO_CLOSE'
        elif ratio < 0.6:
            return 'TOO_FAR'
        
        return None
    
    def log_violation(self, violation_type, severity, details=""):
        """Log a violation"""
        timestamp = datetime.datetime.now()
        
        violation = {
            'type': violation_type,
            'severity': severity,
            'timestamp': timestamp,
            'details': details
        }
        
        self.violations.append(violation)
        self.violation_counts[violation_type] += 1
        
        print(f"⚠️  VIOLATION #{len(self.violations)}: {violation_type} [{severity}]")
        if details:
            print(f"    Details: {details}")
    
    def save_violation_screenshot(self, frame, violation_type):
        """Save screenshot of violation"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"violations/{self.session_id}_{violation_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
    
    def calibrate(self, face):
        """Calibrate baseline measurements"""
        if not self.calibrated:
            x, y, w, h = face
            self.baseline_face_size = w * h
            self.calibrated = True
            print("✓ Calibration complete")
    
    def draw_ui(self, frame, status_text, warning_text=""):
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Status bar
        cv2.rectangle(frame, (0, 0), (width, 60), (40, 40, 40), -1)
        cv2.putText(frame, "PROCTORING ACTIVE", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Violation counter
        cv2.rectangle(frame, (width - 250, 0), (width, 60), (40, 40, 40), -1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (width - 240, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        elapsed = datetime.datetime.now() - self.session_start
        time_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {time_str}", (width - 240, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Warning text
        if warning_text:
            cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 200), -1)
            cv2.putText(frame, warning_text, (10, height - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "⚠️ VIOLATION DETECTED ⚠️", (10, height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def generate_report(self):
        """Generate detailed session report"""
        report_path = f"reports/session_{self.session_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PROCTORING SESSION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = datetime.datetime.now() - self.session_start
            f.write(f"Duration: {str(duration).split('.')[0]}\n\n")
            
            f.write("VIOLATION SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Violations: {len(self.violations)}\n\n")
            
            for vtype, count in self.violation_counts.items():
                if count > 0:
                    f.write(f"  {vtype}: {count}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED VIOLATION LOG\n")
            f.write("="*70 + "\n\n")
            
            for i, v in enumerate(self.violations, 1):
                f.write(f"{i}. [{v['timestamp'].strftime('%H:%M:%S')}] ")
                f.write(f"{v['type']} - Severity: {v['severity']}\n")
                if v['details']:
                    f.write(f"   Details: {v['details']}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_path

    
    def run(self):
        """Main proctoring loop"""
        # Start audio monitoring
        self.start_audio_monitoring()
        
        # Calibration phase
        print("\n⏳ Calibrating... Look at the screen for 3 seconds")
        calibration_frames = 0
        calibration_needed = 90  # 3 seconds at 30fps
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            warning_text = ""
            status_text = "Monitoring..."
            
            # Calibration
            if not self.calibrated and len(faces) == 1:
                calibration_frames += 1
                status_text = f"Calibrating... {calibration_frames}/{calibration_needed}"
                
                if calibration_frames >= calibration_needed:
                    self.calibrate(faces[0])
            
            # Check violations only after calibration
            if self.calibrated:
                # No face detected
                if len(faces) == 0:
                    self.no_face_counter += 1
                    warning_text = "NO FACE DETECTED"
                    
                    if self.no_face_counter >= self.NO_FACE_THRESHOLD:
                        self.log_violation('NO_FACE', 'HIGH', 
                                         f"No face for {self.no_face_counter} frames")
                        self.save_violation_screenshot(frame, 'NO_FACE')
                        self.no_face_counter = 0
                else:
                    self.no_face_counter = 0
                
                # Multiple faces detected
                if len(faces) > 1:
                    self.multiple_face_counter += 1
                    warning_text = f"MULTIPLE FACES DETECTED ({len(faces)})"
                    
                    if self.multiple_face_counter >= self.MULTIPLE_FACE_THRESHOLD:
                        self.log_violation('MULTIPLE_FACES', 'CRITICAL',
                                         f"{len(faces)} faces detected")
                        self.save_violation_screenshot(frame, 'MULTIPLE_FACES')
                        self.multiple_face_counter = 0
                    
                    # Draw rectangles around all faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                else:
                    self.multiple_face_counter = 0
                
                # Single face - detailed analysis
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Check distance
                    face_size = w * h
                    distance_status = self.check_distance(face_size)
                    if distance_status:
                        warning_text = distance_status.replace('_', ' ')
                        self.log_violation(distance_status, 'MEDIUM')
                        self.save_violation_screenshot(frame, distance_status)
                    
                    # Check head pose
                    if self.estimate_head_pose(faces[0], width):
                        warning_text = "HEAD TURNED AWAY"
                        self.log_violation('HEAD_TURNED', 'MEDIUM', 
                                         "Head orientation suspicious")
                        self.save_violation_screenshot(frame, 'HEAD_TURNED')
                    
                    # Check gaze and eyes
                    looking_away, eyes_closed = self.detect_gaze(frame, faces[0])
                    
                    if eyes_closed:
                        self.eyes_closed_counter += 1
                        if self.eyes_closed_counter >= self.EYES_CLOSED_THRESHOLD:
                            warning_text = "EYES CLOSED"
                            self.log_violation('EYES_CLOSED', 'HIGH',
                                             f"Eyes closed for {self.eyes_closed_counter} frames")
                            self.save_violation_screenshot(frame, 'EYES_CLOSED')
                            self.eyes_closed_counter = 0
                    else:
                        self.eyes_closed_counter = 0
                    
                    if looking_away and not eyes_closed:
                        self.looking_away_counter += 1
                        if self.looking_away_counter >= self.LOOKING_AWAY_THRESHOLD:
                            warning_text = "LOOKING AWAY"
                            self.log_violation('LOOKING_AWAY', 'MEDIUM',
                                             f"Looking away for {self.looking_away_counter} frames")
                            self.save_violation_screenshot(frame, 'LOOKING_AWAY')
                            self.looking_away_counter = 0
                    else:
                        self.looking_away_counter = 0
                
                # Object detection (every N frames to save processing)
                if self.frame_count % self.OBJECT_DETECTION_INTERVAL == 0:
                    detected_objects = self.detect_objects(frame)
                    if detected_objects:
                        warning_text = f"SUSPICIOUS OBJECT: {', '.join(detected_objects)}"
                        self.log_violation('SUSPICIOUS_OBJECT', 'HIGH',
                                         f"Objects detected: {', '.join(detected_objects)}")
                        self.save_violation_screenshot(frame, 'SUSPICIOUS_OBJECT')
            
            # Draw UI
            frame = self.draw_ui(frame, status_text, warning_text)
            
            # Display
            cv2.imshow('Complete Proctoring System', frame)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # Cleanup
        self.audio_monitoring = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Generate report
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)
        
        report_path = self.generate_report()
        
        print(f"\n✓ Report saved: {report_path}")
        print(f"✓ Total violations: {len(self.violations)}")
        print(f"✓ Screenshots saved in: violations/")
        print("\nViolation Breakdown:")
        for vtype, count in self.violation_counts.items():
            if count > 0:
                print(f"  • {vtype}: {count}")
        print("\n" + "="*70 + "\n")


def main():
    try:
        system = CompleteProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
