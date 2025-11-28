"""
FULL AI PROCTORING SYSTEM
Features:
- Face detection (no face, multiple faces)
- Gaze tracking (looking away)
- Eye closure detection
- Distance monitoring (too close/too far)
- Head pose estimation
- Real-time alerts and comprehensive logging
"""

import cv2
import numpy as np
import os
import datetime
import time

class FullProctoringSystem:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
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
            'PROFILE_DETECTED': 0
        }
        
        # Thresholds (in frames)
        self.NO_FACE_THRESHOLD = 30  # ~1 second
        self.MULTIPLE_FACE_THRESHOLD = 15  # ~0.5 seconds
        self.LOOKING_AWAY_THRESHOLD = 45  # ~1.5 seconds
        self.EYES_CLOSED_THRESHOLD = 60  # ~2 seconds
        self.HEAD_TURNED_THRESHOLD = 30  # ~1 second
        
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
        
        # Alert sound (Windows beep)
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts
        
        print("\n" + "="*70)
        print("FULL AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ Video monitoring initialized")
        print("✓ Multi-level violation detection enabled")
        print("\nDetection Features:")
        print("  • Face detection (single person only)")
        print("  • Multiple face detection")
        print("  • Gaze tracking (looking away)")
        print("  • Eye closure detection (sleeping)")
        print("  • Distance monitoring (too close/too far)")
        print("  • Head pose estimation (turned away)")
        print("  • Profile face detection (looking sideways)")
        print("\nInstructions:")
        print("  - Sit 2-3 feet from camera")
        print("  - Ensure good lighting on your face")
        print("  - Look at screen during calibration")
        print("  - Press ESC to end session")
        print("="*70 + "\n")
    
    def play_alert(self):
        """Play alert sound (Windows only)"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                import winsound
                winsound.Beep(1000, 200)  # 1000 Hz for 200ms
            except:
                pass
            self.last_alert_time = current_time
    
    def detect_gaze(self, frame, face):
        """Detect if person is looking away"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
        
        # Draw detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # No eyes detected - either closed or looking away
        if len(eyes) == 0:
            return True, True  # Assume eyes closed
        
        # Only one eye detected - likely looking away
        if len(eyes) == 1:
            return True, False
        
        # Check eye positions relative to face center
        if len(eyes) >= 2:
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_center_x = ex + ew // 2
                eye_centers.append(eye_center_x)
            
            face_center = w // 2
            eyes_center = sum(eye_centers) / 2
            deviation = abs(eyes_center - face_center)
            
            # If eyes are significantly off-center, person is looking away
            if deviation > w * 0.25:
                return True, False
        
        return False, False
    
    def estimate_head_pose(self, face, frame_width):
        """Estimate if head is turned away"""
        x, y, w, h = face
        face_center_x = x + w // 2
        frame_center_x = frame_width // 2
        
        # Check horizontal position
        deviation = abs(face_center_x - frame_center_x)
        if deviation > frame_width * 0.3:
            return True
        
        # Check face aspect ratio (turned faces are narrower)
        aspect_ratio = w / h
        if aspect_ratio < 0.65:  # Face too narrow - turned away
            return True
        
        # Check if face width is significantly smaller than baseline
        if self.baseline_face_width:
            width_ratio = w / self.baseline_face_width
            if width_ratio < 0.7:  # Face appears narrower
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
        
        # Play alert for high severity violations
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

    
    def draw_ui(self, frame, status_text, warning_text="", severity=""):
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
            # Color based on severity
            if severity == 'CRITICAL':
                color = (0, 0, 255)  # Red
            elif severity == 'HIGH':
                color = (0, 100, 255)  # Orange
            else:
                color = (0, 165, 255)  # Yellow
            
            cv2.rectangle(frame, (0, height - 90), (width, height), color, -1)
            cv2.putText(frame, "⚠️ VIOLATION DETECTED ⚠️", (width//2 - 200, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, warning_text, (width//2 - len(warning_text)*8, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Violation summary (left side)
        y_offset = 90
        cv2.rectangle(frame, (0, y_offset), (250, y_offset + 200), (40, 40, 40), -1)
        cv2.putText(frame, "Violation Summary:", (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos = y_offset + 50
        for vtype, count in self.violation_counts.items():
            if count > 0:
                cv2.putText(frame, f"{vtype}: {count}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 20
        
        return frame
    
    def generate_report(self):
        """Generate detailed session report"""
        report_path = f"reports/session_{self.session_id}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FULL PROCTORING SESSION REPORT\n")
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
            for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    f.write(f"  {vtype:20s}: {count:3d}\n")
            
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
            
            # Simple assessment
            total_violations = len(self.violations)
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
            
            if critical > 0:
                f.write("\n⚠️  WARNING: Critical violations detected (multiple faces)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_path
    
    def run(self):
        """Main proctoring loop"""
        # Calibration phase
        print("⏳ Calibrating... Look at the screen for 3 seconds\n")
        calibration_frames = 0
        calibration_needed = 90  # 3 seconds at 30fps
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture video")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            
            # Detect frontal faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            # Detect profile faces (looking sideways)
            profiles = self.profile_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            warning_text = ""
            severity = ""
            status_text = "Monitoring..."
            
            # Calibration phase
            if not self.calibrated and len(faces) == 1:
                calibration_frames += 1
                status_text = f"Calibrating... {calibration_frames}/{calibration_needed}"
                
                # Draw calibration progress
                progress = int((calibration_frames / calibration_needed) * width)
                cv2.rectangle(frame, (0, height - 20), (progress, height), (0, 255, 0), -1)
                
                if calibration_frames >= calibration_needed:
                    self.calibrate(faces[0])
            
            # Monitoring phase (after calibration)
            if self.calibrated:
                # NO FACE DETECTED
                if len(faces) == 0 and len(profiles) == 0:
                    self.no_face_counter += 1
                    warning_text = "NO FACE DETECTED - Return to camera view"
                    severity = "HIGH"
                    
                    if self.no_face_counter >= self.NO_FACE_THRESHOLD:
                        self.log_violation('NO_FACE', 'HIGH', 
                                         f"No face detected for {self.no_face_counter} frames")
                        self.save_violation_screenshot(frame, 'NO_FACE')
                        self.no_face_counter = 0
                else:
                    self.no_face_counter = 0
                
                # PROFILE DETECTED (looking sideways)
                if len(profiles) > 0 and len(faces) == 0:
                    warning_text = "PROFILE DETECTED - Face the camera"
                    severity = "MEDIUM"
                    self.log_violation('PROFILE_DETECTED', 'MEDIUM', 
                                     "Profile face detected - looking sideways")
                    self.save_violation_screenshot(frame, 'PROFILE_DETECTED')
                    
                    # Draw profile rectangles
                    for (x, y, w, h) in profiles:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                
                # MULTIPLE FACES DETECTED
                if len(faces) > 1:
                    self.multiple_face_counter += 1
                    warning_text = f"MULTIPLE FACES DETECTED ({len(faces)}) - Only one person allowed"
                    severity = "CRITICAL"
                    
                    if self.multiple_face_counter >= self.MULTIPLE_FACE_THRESHOLD:
                        self.log_violation('MULTIPLE_FACES', 'CRITICAL',
                                         f"{len(faces)} faces detected")
                        self.save_violation_screenshot(frame, 'MULTIPLE_FACES')
                        self.multiple_face_counter = 0
                    
                    # Draw rectangles around all faces
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
                    if distance_status:
                        warning_text = distance_status.replace('_', ' ')
                        severity = "MEDIUM"
                        self.log_violation(distance_status, 'MEDIUM',
                                         f"Face size ratio: {face_size/self.baseline_face_size:.2f}")
                        self.save_violation_screenshot(frame, distance_status)
                    
                    # Head pose check
                    if self.estimate_head_pose(faces[0], width):
                        self.head_turned_counter += 1
                        if self.head_turned_counter >= self.HEAD_TURNED_THRESHOLD:
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
                            warning_text = "LOOKING AWAY FROM SCREEN"
                            severity = "MEDIUM"
                            self.log_violation('LOOKING_AWAY', 'MEDIUM',
                                             f"Looking away for {self.looking_away_counter} frames (~{self.looking_away_counter//30}s)")
                            self.save_violation_screenshot(frame, 'LOOKING_AWAY')
                            self.looking_away_counter = 0
                    else:
                        self.looking_away_counter = 0
            
            # Draw UI
            frame = self.draw_ui(frame, status_text, warning_text, severity)
            
            # Display
            cv2.imshow('Full Proctoring System - Press ESC to Exit', frame)
            
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
            for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  • {vtype}: {count}")
        else:
            print("\n✓ No violations detected - Excellent session!")
        
        print("\n" + "="*70 + "\n")


def main():
    try:
        system = FullProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
