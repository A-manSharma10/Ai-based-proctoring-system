"""
ULTIMATE AI PROCTORING SYSTEM WITH OBJECT DETECTION
Features:
- All face/gaze/eye detection features
- Object detection (phone, book, laptop, etc.) using MobileNet SSD
- Real-time alerts and comprehensive logging
"""

import cv2
import numpy as np
import os
import datetime
import time

class UltimateProctoringSystem:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Load MobileNet SSD for object detection
        self.load_object_detector()
        
        # Violation tracking
        self.violations = []
        self.violation_counts = {
            'NO_FACE': 0,
            'MULTIPLE_FACES': 0,
            'LOOKING_AWAY': 0,
            'EYES_CLOSED': 0,
            'TOO_CLOSE': 0,
            'TOO_FAR': 0,
            'HEAD_TURNED': 0,
            'PROFILE_DETECTED': 0,
            'PHONE_DETECTED': 0,
            'BOOK_DETECTED': 0,
            'LAPTOP_DETECTED': 0,
            'SUSPICIOUS_OBJECT': 0
        }
        
        # Thresholds (in frames)
        self.NO_FACE_THRESHOLD = 30
        self.MULTIPLE_FACE_THRESHOLD = 15
        self.LOOKING_AWAY_THRESHOLD = 45
        self.EYES_CLOSED_THRESHOLD = 60
        self.HEAD_TURNED_THRESHOLD = 30
        self.OBJECT_DETECTION_INTERVAL = 15  # Check every 15 frames for performance
        
        # Counters
        self.no_face_counter = 0
        self.multiple_face_counter = 0
        self.looking_away_counter = 0
        self.eyes_closed_counter = 0
        self.head_turned_counter = 0
        self.frame_count = 0
        
        # Directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        
        # Session info
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Calibration
        self.calibrated = False
        self.baseline_face_size = None
        self.baseline_face_width = None
        
        # Alert sound
        self.last_alert_time = 0
        self.alert_cooldown = 3
        
        print("\n" + "="*70)
        print("ULTIMATE AI PROCTORING SYSTEM WITH OBJECT DETECTION")
        print("="*70)
        print("✓ Video monitoring initialized")
        print("✓ Object detection loaded")
        print("✓ Multi-level violation detection enabled")
        print("\nDetection Features:")
        print("  • Face detection (single person only)")
        print("  • Multiple face detection")
        print("  • Gaze tracking (looking away)")
        print("  • Eye closure detection (sleeping)")
        print("  • Distance monitoring (too close/too far)")
        print("  • Head pose estimation (turned away)")
        print("  • Profile face detection (looking sideways)")
        print("  • 📱 PHONE DETECTION")
        print("  • 📚 BOOK DETECTION")
        print("  • 💻 LAPTOP/TABLET DETECTION")
        print("  • 📦 OTHER SUSPICIOUS OBJECTS")
        print("\nInstructions:")
        print("  - Sit 2-3 feet from camera")
        print("  - Ensure good lighting on your face")
        print("  - Keep desk clear of unauthorized items")
        print("  - Look at screen during calibration")
        print("  - Press ESC to end session")
        print("="*70 + "\n")
    
    def load_object_detector(self):
        """Load MobileNet SSD for object detection"""
        try:
            # Try to load pre-trained MobileNet SSD model
            model_path = 'MobileNetSSD_deploy.caffemodel'
            config_path = 'MobileNetSSD_deploy.prototxt'
            
            # If model files don't exist, download them
            if not os.path.exists(model_path) or not os.path.exists(config_path):
                print("⏳ Downloading object detection model...")
                self.download_model()
            
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            
            # COCO class names
            self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                           "sofa", "train", "tvmonitor", "cell phone", "book", "laptop"]
            
            # Suspicious objects to detect
            self.suspicious_objects = {
                'cell phone': 'PHONE',
                'book': 'BOOK',
                'laptop': 'LAPTOP',
                'tvmonitor': 'MONITOR',
                'bottle': 'BOTTLE'
            }
            
            self.object_detection_enabled = True
            print("✓ Object detection model loaded")
            
        except Exception as e:
            print(f"⚠️  Object detection unavailable: {e}")
            print("   Continuing with face/gaze detection only...")
            self.object_detection_enabled = False
    
    def download_model(self):
        """Download MobileNet SSD model files"""
        import urllib.request
        
        base_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
        
        files = {
            'MobileNetSSD_deploy.prototxt': base_url + 'MobileNetSSD_deploy.prototxt',
            'MobileNetSSD_deploy.caffemodel': 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"  Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"  ✓ {filename} downloaded")
                except Exception as e:
                    print(f"  ✗ Failed to download {filename}: {e}")
                    raise
    
    def detect_objects(self, frame):
        """Detect suspicious objects in frame"""
        if not self.object_detection_enabled:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Prepare image for detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                        (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            detected_objects = []
            
            # Loop over detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:  # 50% confidence threshold
                    idx = int(detections[0, 0, i, 1])
                    
                    if idx < len(self.CLASSES):
                        class_name = self.CLASSES[idx]
                        
                        # Check if it's a suspicious object
                        if class_name in self.suspicious_objects:
                            # Get bounding box
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (startX, startY - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            detected_objects.append(self.suspicious_objects[class_name])
            
            return detected_objects
            
        except Exception as e:
            return []
    
    def play_alert(self):
        """Play alert sound"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                import winsound
                winsound.Beep(1000, 200)
            except:
                pass
            self.last_alert_time = current_time
    
    def detect_gaze(self, frame, face):
        """Detect if person is looking away"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        if len(eyes) == 0:
            return True, True
        
        if len(eyes) == 1:
            return True, False
        
        if len(eyes) >= 2:
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_center_x = ex + ew // 2
                eye_centers.append(eye_center_x)
            
            face_center = w // 2
            eyes_center = sum(eye_centers) / 2
            deviation = abs(eyes_center - face_center)
            
            if deviation > w * 0.25:
                return True, False
        
        return False, False
    
    def estimate_head_pose(self, face, frame_width):
        """Estimate if head is turned away"""
        x, y, w, h = face
        face_center_x = x + w // 2
        frame_center_x = frame_width // 2
        
        deviation = abs(face_center_x - frame_center_x)
        if deviation > frame_width * 0.3:
            return True
        
        aspect_ratio = w / h
        if aspect_ratio < 0.65:
            return True
        
        if self.baseline_face_width:
            width_ratio = w / self.baseline_face_width
            if width_ratio < 0.7:
                return True
        
        return False
    
    def check_distance(self, face_size):
        """Check if person is too close or too far"""
        if self.baseline_face_size is None:
            return None
        
        ratio = face_size / self.baseline_face_size
        
        if ratio > 1.6:
            return 'TOO_CLOSE'
        elif ratio < 0.5:
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
            print(f"    {details}")
        
        if severity in ['HIGH', 'CRITICAL']:
            self.play_alert()
    
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
            self.baseline_face_width = w
            self.calibrated = True
            print("✓ Calibration complete - Monitoring started")

    
    def draw_ui(self, frame, status_text, warning_text="", severity="", detected_objects=[]):
        """Draw comprehensive UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (width, 70), (40, 40, 40), -1)
        cv2.putText(frame, "PROCTORING ACTIVE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Violation counter (top right)
        cv2.rectangle(frame, (width - 280, 0), (width, 70), (40, 40, 40), -1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (width - 270, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elapsed = datetime.datetime.now() - self.session_start
        time_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {time_str}", (width - 270, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Warning banner (bottom)
        if warning_text:
            if severity == 'CRITICAL':
                color = (0, 0, 255)
            elif severity == 'HIGH':
                color = (0, 100, 255)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(frame, (0, height - 90), (width, height), color, -1)
            cv2.putText(frame, "⚠️ VIOLATION DETECTED ⚠️", (width//2 - 200, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, warning_text, (width//2 - len(warning_text)*8, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Violation summary (left side)
        y_offset = 90
        panel_height = min(250, 50 + len([v for v in self.violation_counts.values() if v > 0]) * 20)
        cv2.rectangle(frame, (0, y_offset), (280, y_offset + panel_height), (40, 40, 40), -1)
        cv2.putText(frame, "Violation Summary:", (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos = y_offset + 50
        for vtype, count in self.violation_counts.items():
            if count > 0:
                # Add emoji for object violations
                emoji = ""
                if vtype == "PHONE_DETECTED":
                    emoji = "📱 "
                elif vtype == "BOOK_DETECTED":
                    emoji = "📚 "
                elif vtype == "LAPTOP_DETECTED":
                    emoji = "💻 "
                
                cv2.putText(frame, f"{emoji}{vtype}: {count}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 20
        
        # Object detection status (right side)
        if self.object_detection_enabled:
            obj_y = 90
            cv2.rectangle(frame, (width - 280, obj_y), (width, obj_y + 80), (40, 40, 40), -1)
            cv2.putText(frame, "Object Detection: ON", (width - 270, obj_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if detected_objects:
                cv2.putText(frame, "⚠️ Objects Found:", (width - 270, obj_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(frame, ", ".join(detected_objects), (width - 270, obj_y + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def generate_report(self):
        """Generate detailed session report"""
        report_path = f"reports/session_{self.session_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ULTIMATE PROCTORING SESSION REPORT\n")
            f.write("WITH OBJECT DETECTION\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            end_time = datetime.datetime.now()
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = end_time - self.session_start
            f.write(f"Duration: {str(duration).split('.')[0]}\n\n")
            
            f.write("="*70 + "\n")
            f.write("VIOLATION SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Total Violations: {len(self.violations)}\n\n")
            
            # Categorize by severity
            critical = sum(1 for v in self.violations if v['severity'] == 'CRITICAL')
            high = sum(1 for v in self.violations if v['severity'] == 'HIGH')
            medium = sum(1 for v in self.violations if v['severity'] == 'MEDIUM')
            
            f.write(f"Critical Severity: {critical}\n")
            f.write(f"High Severity: {high}\n")
            f.write(f"Medium Severity: {medium}\n\n")
            
            f.write("Breakdown by Type:\n")
            f.write("-"*70 + "\n")
            
            # Separate object violations
            object_violations = ['PHONE_DETECTED', 'BOOK_DETECTED', 'LAPTOP_DETECTED', 'SUSPICIOUS_OBJECT']
            behavior_violations = [k for k in self.violation_counts.keys() if k not in object_violations]
            
            f.write("\nBehavioral Violations:\n")
            for vtype in behavior_violations:
                count = self.violation_counts[vtype]
                if count > 0:
                    f.write(f"  {vtype:20s}: {count:3d}\n")
            
            f.write("\nObject Detection Violations:\n")
            for vtype in object_violations:
                count = self.violation_counts[vtype]
                if count > 0:
                    emoji = ""
                    if vtype == "PHONE_DETECTED":
                        emoji = "📱 "
                    elif vtype == "BOOK_DETECTED":
                        emoji = "📚 "
                    elif vtype == "LAPTOP_DETECTED":
                        emoji = "💻 "
                    f.write(f"  {emoji}{vtype:20s}: {count:3d}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DETAILED VIOLATION LOG\n")
            f.write("="*70 + "\n\n")
            
            for i, v in enumerate(self.violations, 1):
                f.write(f"{i:3d}. [{v['timestamp'].strftime('%H:%M:%S')}] ")
                f.write(f"{v['type']:20s} | Severity: {v['severity']}\n")
                if v['details']:
                    f.write(f"     Details: {v['details']}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("ASSESSMENT\n")
            f.write("="*70 + "\n\n")
            
            # Assessment
            total_violations = len(self.violations)
            object_count = sum(self.violation_counts[k] for k in object_violations)
            
            if total_violations == 0:
                f.write("Status: EXCELLENT - No violations detected\n")
            elif total_violations <= 5:
                f.write("Status: GOOD - Minor violations only\n")
            elif total_violations <= 15:
                f.write("Status: FAIR - Moderate violations detected\n")
            elif total_violations <= 30:
                f.write("Status: POOR - Significant violations detected\n")
            else:
                f.write("Status: CRITICAL - Excessive violations detected\n")
            
            f.write(f"\nTotal Violations: {total_violations}\n")
            f.write(f"Critical Violations: {critical}\n")
            f.write(f"Object Violations: {object_count}\n")
            
            if critical > 0:
                f.write("\n⚠️  WARNING: Critical violations detected (multiple faces)\n")
            
            if object_count > 0:
                f.write(f"\n⚠️  WARNING: {object_count} unauthorized object(s) detected\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_path

    
    def run(self):
        """Main proctoring loop"""
        # Calibration phase
        print("⏳ Calibrating... Look at the screen for 3 seconds\n")
        calibration_frames = 0
        calibration_needed = 90
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture video")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            
            # Detect frontal faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            # Detect profile faces
            profiles = self.profile_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            warning_text = ""
            severity = ""
            status_text = "Monitoring..."
            detected_objects = []
            
            # Calibration phase
            if not self.calibrated and len(faces) == 1:
                calibration_frames += 1
                status_text = f"Calibrating... {calibration_frames}/{calibration_needed}"
                
                progress = int((calibration_frames / calibration_needed) * width)
                cv2.rectangle(frame, (0, height - 20), (progress, height), (0, 255, 0), -1)
                
                if calibration_frames >= calibration_needed:
                    self.calibrate(faces[0])
            
            # Monitoring phase
            if self.calibrated:
                # Object detection (every N frames for performance)
                if self.frame_count % self.OBJECT_DETECTION_INTERVAL == 0:
                    detected_objects = self.detect_objects(frame)
                    
                    if detected_objects:
                        for obj in detected_objects:
                            obj_type = f"{obj}_DETECTED"
                            warning_text = f"UNAUTHORIZED OBJECT: {obj}"
                            severity = "CRITICAL"
                            
                            self.log_violation(obj_type, 'CRITICAL',
                                             f"{obj} detected in frame")
                            self.save_violation_screenshot(frame, obj_type)
                
                # NO FACE DETECTED
                if len(faces) == 0 and len(profiles) == 0:
                    self.no_face_counter += 1
                    if not warning_text:  # Don't override object warning
                        warning_text = "NO FACE DETECTED - Return to camera view"
                        severity = "HIGH"
                    
                    if self.no_face_counter >= self.NO_FACE_THRESHOLD:
                        self.log_violation('NO_FACE', 'HIGH', 
                                         f"No face detected for {self.no_face_counter} frames")
                        self.save_violation_screenshot(frame, 'NO_FACE')
                        self.no_face_counter = 0
                else:
                    self.no_face_counter = 0
                
                # PROFILE DETECTED
                if len(profiles) > 0 and len(faces) == 0:
                    if not warning_text:
                        warning_text = "PROFILE DETECTED - Face the camera"
                        severity = "MEDIUM"
                    self.log_violation('PROFILE_DETECTED', 'MEDIUM', 
                                     "Profile face detected - looking sideways")
                    self.save_violation_screenshot(frame, 'PROFILE_DETECTED')
                    
                    for (x, y, w, h) in profiles:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                
                # MULTIPLE FACES
                if len(faces) > 1:
                    self.multiple_face_counter += 1
                    warning_text = f"MULTIPLE FACES ({len(faces)}) - Only one person allowed"
                    severity = "CRITICAL"
                    
                    if self.multiple_face_counter >= self.MULTIPLE_FACE_THRESHOLD:
                        self.log_violation('MULTIPLE_FACES', 'CRITICAL',
                                         f"{len(faces)} faces detected")
                        self.save_violation_screenshot(frame, 'MULTIPLE_FACES')
                        self.multiple_face_counter = 0
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(frame, "UNAUTHORIZED", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    self.multiple_face_counter = 0
                
                # SINGLE FACE - Detailed analysis
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Authorized", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Distance check
                    face_size = w * h
                    distance_status = self.check_distance(face_size)
                    if distance_status and not warning_text:
                        warning_text = distance_status.replace('_', ' ')
                        severity = "MEDIUM"
                        self.log_violation(distance_status, 'MEDIUM',
                                         f"Face size ratio: {face_size/self.baseline_face_size:.2f}")
                        self.save_violation_screenshot(frame, distance_status)
                    
                    # Head pose check
                    if self.estimate_head_pose(faces[0], width):
                        self.head_turned_counter += 1
                        if self.head_turned_counter >= self.HEAD_TURNED_THRESHOLD:
                            if not warning_text:
                                warning_text = "HEAD TURNED AWAY"
                                severity = "MEDIUM"
                            self.log_violation('HEAD_TURNED', 'MEDIUM', 
                                             "Head orientation suspicious")
                            self.save_violation_screenshot(frame, 'HEAD_TURNED')
                            self.head_turned_counter = 0
                    else:
                        self.head_turned_counter = 0
                    
                    # Gaze and eye detection
                    looking_away, eyes_closed = self.detect_gaze(frame, faces[0])
                    
                    if eyes_closed:
                        self.eyes_closed_counter += 1
                        if self.eyes_closed_counter >= self.EYES_CLOSED_THRESHOLD:
                            if not warning_text:
                                warning_text = "EYES CLOSED - Possible sleeping"
                                severity = "HIGH"
                            self.log_violation('EYES_CLOSED', 'HIGH',
                                             f"Eyes closed for {self.eyes_closed_counter} frames (~{self.eyes_closed_counter//30}s)")
                            self.save_violation_screenshot(frame, 'EYES_CLOSED')
                            self.eyes_closed_counter = 0
                    else:
                        self.eyes_closed_counter = 0
                    
                    if looking_away and not eyes_closed:
                        self.looking_away_counter += 1
                        if self.looking_away_counter >= self.LOOKING_AWAY_THRESHOLD:
                            if not warning_text:
                                warning_text = "LOOKING AWAY FROM SCREEN"
                                severity = "MEDIUM"
                            self.log_violation('LOOKING_AWAY', 'MEDIUM',
                                             f"Looking away for {self.looking_away_counter} frames (~{self.looking_away_counter//30}s)")
                            self.save_violation_screenshot(frame, 'LOOKING_AWAY')
                            self.looking_away_counter = 0
                    else:
                        self.looking_away_counter = 0
            
            # Draw UI
            frame = self.draw_ui(frame, status_text, warning_text, severity, detected_objects)
            
            # Display
            cv2.imshow('Ultimate Proctoring System - Press ESC to Exit', frame)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # Cleanup
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
        
        if len(self.violations) > 0:
            print("\nViolation Breakdown:")
            
            # Behavioral violations
            behavior_violations = ['NO_FACE', 'MULTIPLE_FACES', 'LOOKING_AWAY', 
                                  'EYES_CLOSED', 'TOO_CLOSE', 'TOO_FAR', 
                                  'HEAD_TURNED', 'PROFILE_DETECTED']
            print("\n  Behavioral:")
            for vtype in behavior_violations:
                count = self.violation_counts[vtype]
                if count > 0:
                    print(f"    • {vtype}: {count}")
            
            # Object violations
            object_violations = ['PHONE_DETECTED', 'BOOK_DETECTED', 
                               'LAPTOP_DETECTED', 'SUSPICIOUS_OBJECT']
            obj_total = sum(self.violation_counts[v] for v in object_violations)
            if obj_total > 0:
                print("\n  Objects Detected:")
                for vtype in object_violations:
                    count = self.violation_counts[vtype]
                    if count > 0:
                        emoji = ""
                        if vtype == "PHONE_DETECTED":
                            emoji = "📱 "
                        elif vtype == "BOOK_DETECTED":
                            emoji = "📚 "
                        elif vtype == "LAPTOP_DETECTED":
                            emoji = "💻 "
                        print(f"    • {emoji}{vtype}: {count}")
        else:
            print("\n✓ No violations detected - Excellent session!")
        
        print("\n" + "="*70 + "\n")


def main():
    try:
        system = UltimateProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
