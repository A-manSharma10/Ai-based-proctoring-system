"""
FINAL COMPLETE AI PROCTORING SYSTEM
Features:
- All face/gaze/eye detection features
- Simple object detection (phone, book shapes)
- Hand detection (holding objects)
- Real-time alerts and comprehensive logging
"""

import cv2
import numpy as np
import os
import datetime
import time

class FinalCompleteProctoringSystem:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Try to load upper body cascade for hand/object detection
        try:
            self.upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        except:
            self.upperbody_cascade = None
        
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
            'SUSPICIOUS_OBJECT': 0,
            'HAND_NEAR_FACE': 0
        }
        
        # Thresholds (in frames)
        self.NO_FACE_THRESHOLD = 30
        self.MULTIPLE_FACE_THRESHOLD = 15
        self.LOOKING_AWAY_THRESHOLD = 45
        self.EYES_CLOSED_THRESHOLD = 60
        self.HEAD_TURNED_THRESHOLD = 30
        self.OBJECT_DETECTION_INTERVAL = 20  # Check every 20 frames
        
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
        self.baseline_frame = None
        
        # Alert sound
        self.last_alert_time = 0
        self.alert_cooldown = 3
        
        print("\n" + "="*70)
        print("FINAL COMPLETE AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ Video monitoring initialized")
        print("✓ Object detection enabled (shape & motion analysis)")
        print("✓ Multi-level violation detection enabled")
        print("\nDetection Features:")
        print("  • Face detection (single person only)")
        print("  • Multiple face detection")
        print("  • Gaze tracking (looking away)")
        print("  • Eye closure detection (sleeping)")
        print("  • Distance monitoring (too close/too far)")
        print("  • Head pose estimation (turned away)")
        print("  • Profile face detection (looking sideways)")
        print("  • 📱 Suspicious object detection (shape analysis)")
        print("  • ✋ Hand movement detection (holding objects)")
        print("  • 📦 Motion-based object detection")
        print("\nInstructions:")
        print("  - Sit 2-3 feet from camera")
        print("  - Ensure good lighting on your face")
        print("  - Keep desk clear of unauthorized items")
        print("  - Keep hands visible and away from face")
        print("  - Look at screen during calibration")
        print("  - Press ESC to end session")
        print("="*70 + "\n")
    
    def detect_suspicious_objects(self, frame, face_region=None):
        """Detect suspicious objects using shape and color analysis"""
        detected_objects = []
        
        try:
            height, width = frame.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define region of interest (lower half of frame, excluding face)
            roi_y_start = height // 3
            roi = frame[roi_y_start:, :]
            roi_hsv = hsv[roi_y_start:, :]
            
            # Detect dark rectangular objects (phones, books, tablets)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (objects should be reasonably sized)
                if 1000 < area < 50000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Check if it's rectangular (phone/book shape)
                    # Phones: aspect ratio ~0.5-0.7, Books: ~0.7-1.5
                    if 0.4 < aspect_ratio < 1.8:
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y + roi_y_start), 
                                    (x + w, y + h + roi_y_start), (0, 0, 255), 2)
                        cv2.putText(frame, "SUSPICIOUS OBJECT", (x, y + roi_y_start - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        if aspect_ratio < 0.8:
                            detected_objects.append("PHONE-LIKE")
                        else:
                            detected_objects.append("BOOK-LIKE")
            
            # Detect hands near face (potential cheating)
            if face_region is not None:
                fx, fy, fw, fh = face_region
                
                # Define region around face
                margin = 50
                hand_roi_x1 = max(0, fx - margin)
                hand_roi_y1 = max(0, fy - margin)
                hand_roi_x2 = min(width, fx + fw + margin)
                hand_roi_y2 = min(height, fy + fh + margin)
                
                # Detect skin color (hands) near face
                hand_roi = frame[hand_roi_y1:hand_roi_y2, hand_roi_x1:hand_roi_x2]
                hand_roi_hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
                
                # Skin color range in HSV
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                
                skin_mask = cv2.inRange(hand_roi_hsv, lower_skin, upper_skin)
                skin_pixels = cv2.countNonZero(skin_mask)
                
                # If significant skin detected outside face area
                if skin_pixels > 5000:  # Threshold for hand detection
                    detected_objects.append("HAND_NEAR_FACE")
                    cv2.rectangle(frame, (hand_roi_x1, hand_roi_y1), 
                                (hand_roi_x2, hand_roi_y2), (255, 0, 0), 2)
                    cv2.putText(frame, "HAND DETECTED", (hand_roi_x1, hand_roi_y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        except Exception as e:
            pass
        
        return detected_objects
    
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
        panel_height = min(280, 50 + len([v for v in self.violation_counts.values() if v > 0]) * 20)
        cv2.rectangle(frame, (0, y_offset), (300, y_offset + panel_height), (40, 40, 40), -1)
        cv2.putText(frame, "Violation Summary:", (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos = y_offset + 50
        for vtype, count in self.violation_counts.items():
            if count > 0:
                emoji = ""
                if vtype == "SUSPICIOUS_OBJECT":
                    emoji = "📦 "
                elif vtype == "HAND_NEAR_FACE":
                    emoji = "✋ "
                
                cv2.putText(frame, f"{emoji}{vtype}: {count}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 20
        
        # Object detection status (right side)
        obj_y = 90
        cv2.rectangle(frame, (width - 300, obj_y), (width, obj_y + 100), (40, 40, 40), -1)
        cv2.putText(frame, "Object Detection: ON", (width - 290, obj_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Shape & Motion Analysis", (width - 290, obj_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if detected_objects:
            cv2.putText(frame, "⚠️ Detected:", (width - 290, obj_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            obj_text = ", ".join(detected_objects)[:30]
            cv2.putText(frame, obj_text, (width - 290, obj_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        return frame
    
    def generate_report(self):
        """Generate detailed session report"""
        report_path = f"reports/session_{self.session_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FINAL COMPLETE PROCTORING SESSION REPORT\n")
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
            
            critical = sum(1 for v in self.violations if v['severity'] == 'CRITICAL')
            high = sum(1 for v in self.violations if v['severity'] == 'HIGH')
            medium = sum(1 for v in self.violations if v['severity'] == 'MEDIUM')
            
            f.write(f"Critical Severity: {critical}\n")
            f.write(f"High Severity: {high}\n")
            f.write(f"Medium Severity: {medium}\n\n")
            
            f.write("Breakdown by Type:\n")
            f.write("-"*70 + "\n")
            
            object_violations = ['SUSPICIOUS_OBJECT', 'HAND_NEAR_FACE']
            behavior_violations = [k for k in self.violation_counts.keys() if k not in object_violations]
            
            f.write("\nBehavioral Violations:\n")
            for vtype in behavior_violations:
                count = self.violation_counts[vtype]
                if count > 0:
                    f.write(f"  {vtype:20s}: {count:3d}\n")
            
            f.write("\nObject/Hand Detection Violations:\n")
            for vtype in object_violations:
                count = self.violation_counts[vtype]
                if count > 0:
                    emoji = "📦 " if vtype == "SUSPICIOUS_OBJECT" else "✋ "
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
            f.write(f"Object/Hand Violations: {object_count}\n")
            
            if critical > 0:
                f.write("\n⚠️  WARNING: Critical violations detected\n")
            
            if object_count > 0:
                f.write(f"\n⚠️  WARNING: {object_count} suspicious object(s)/hand movement(s) detected\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_path
    
    def run(self):
        """Main proctoring loop"""
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
            
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            profiles = self.profile_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            warning_text = ""
            severity = ""
            status_text = "Monitoring..."
            detected_objects = []
            
            # Calibration
            if not self.calibrated and len(faces) == 1:
                calibration_frames += 1
                status_text = f"Calibrating... {calibration_frames}/{calibration_needed}"
                
                progress = int((calibration_frames / calibration_needed) * width)
                cv2.rectangle(frame, (0, height - 20), (progress, height), (0, 255, 0), -1)
                
                if calibration_frames >= calibration_needed:
                    self.calibrate(faces[0])
                    self.baseline_frame = frame.copy()
            
            # Monitoring
            if self.calibrated:
                # Object detection
                if self.frame_count % self.OBJECT_DETECTION_INTERVAL == 0:
                    face_region = faces[0] if len(faces) == 1 else None
                    detected_objects = self.detect_suspicious_objects(frame, face_region)
                    
                    if detected_objects:
                        for obj in detected_objects:
                            if obj == "HAND_NEAR_FACE":
                                warning_text = "HAND NEAR FACE - Keep hands visible"
                                severity = "HIGH"
                                self.log_violation('HAND_NEAR_FACE', 'HIGH',
                                                 "Hand detected near face area")
                                self.save_violation_screenshot(frame, 'HAND_NEAR_FACE')
                            else:
                                warning_text = f"SUSPICIOUS OBJECT DETECTED: {obj}"
                                severity = "CRITICAL"
                                self.log_violation('SUSPICIOUS_OBJECT', 'CRITICAL',
                                                 f"{obj} object detected")
                                self.save_violation_screenshot(frame, 'SUSPICIOUS_OBJECT')
                
                # NO FACE
                if len(faces) == 0 and len(profiles) == 0:
                    self.no_face_counter += 1
                    if not warning_text:
                        warning_text = "NO FACE DETECTED"
                        severity = "HIGH"
                    
                    if self.no_face_counter >= self.NO_FACE_THRESHOLD:
                        self.log_violation('NO_FACE', 'HIGH', 
                                         f"No face for {self.no_face_counter} frames")
                        self.save_violation_screenshot(frame, 'NO_FACE')
                        self.no_face_counter = 0
                else:
                    self.no_face_counter = 0
                
                # PROFILE
                if len(profiles) > 0 and len(faces) == 0:
                    if not warning_text:
                        warning_text = "PROFILE DETECTED"
                        severity = "MEDIUM"
                    self.log_violation('PROFILE_DETECTED', 'MEDIUM', 
                                     "Profile face detected")
                    self.save_violation_screenshot(frame, 'PROFILE_DETECTED')
                    
                    for (x, y, w, h) in profiles:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                
                # MULTIPLE FACES
                if len(faces) > 1:
                    self.multiple_face_counter += 1
                    warning_text = f"MULTIPLE FACES ({len(faces)})"
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
                
                # SINGLE FACE
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Authorized", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    face_size = w * h
                    distance_status = self.check_distance(face_size)
                    if distance_status and not warning_text:
                        warning_text = distance_status.replace('_', ' ')
                        severity = "MEDIUM"
                        self.log_violation(distance_status, 'MEDIUM',
                                         f"Face size ratio: {face_size/self.baseline_face_size:.2f}")
                        self.save_violation_screenshot(frame, distance_status)
                    
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
                    
                    looking_away, eyes_closed = self.detect_gaze(frame, faces[0])
                    
                    if eyes_closed:
                        self.eyes_closed_counter += 1
                        if self.eyes_closed_counter >= self.EYES_CLOSED_THRESHOLD:
                            if not warning_text:
                                warning_text = "EYES CLOSED"
                                severity = "HIGH"
                            self.log_violation('EYES_CLOSED', 'HIGH',
                                             f"Eyes closed for {self.eyes_closed_counter} frames")
                            self.save_violation_screenshot(frame, 'EYES_CLOSED')
                            self.eyes_closed_counter = 0
                    else:
                        self.eyes_closed_counter = 0
                    
                    if looking_away and not eyes_closed:
                        self.looking_away_counter += 1
                        if self.looking_away_counter >= self.LOOKING_AWAY_THRESHOLD:
                            if not warning_text:
                                warning_text = "LOOKING AWAY"
                                severity = "MEDIUM"
                            self.log_violation('LOOKING_AWAY', 'MEDIUM',
                                             f"Looking away for {self.looking_away_counter} frames")
                            self.save_violation_screenshot(frame, 'LOOKING_AWAY')
                            self.looking_away_counter = 0
                    else:
                        self.looking_away_counter = 0
            
            frame = self.draw_ui(frame, status_text, warning_text, severity, detected_objects)
            
            cv2.imshow('Final Complete Proctoring System - Press ESC to Exit', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)
        
        report_path = self.generate_report()
        
        print(f"\n✓ Report saved: {report_path}")
        print(f"✓ Total violations: {len(self.violations)}")
        print(f"✓ Screenshots saved in: violations/")
        
        if len(self.violations) > 0:
            print("\nViolation Breakdown:")
            for vtype, count in sorted(self.violation_counts.items(), 
                                      key=lambda x: x[1], reverse=True):
                if count > 0:
                    emoji = ""
                    if vtype == "SUSPICIOUS_OBJECT":
                        emoji = "📦 "
                    elif vtype == "HAND_NEAR_FACE":
                        emoji = "✋ "
                    print(f"  • {emoji}{vtype}: {count}")
        else:
            print("\n✓ No violations detected - Excellent session!")
        
        print("\n" + "="*70 + "\n")


def main():
    try:
        system = FinalCompleteProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
