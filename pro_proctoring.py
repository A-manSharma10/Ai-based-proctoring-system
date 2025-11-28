"""
PROFESSIONAL AI PROCTORING SYSTEM
Using OpenCV only - No external dependencies needed
Features:
- Enhanced pupil detection
- Real-time gaze tracking
- Distance monitoring (too close/too far)
- Real-time warning messages
- Visual gaze alert bar
- Audio alerts
- Professional UI
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os
import winsound

class ProfessionalProctoring:
    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        os.makedirs('violations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Tracking counters
        self.looking_away_frames = 0
        self.no_face_frames = 0
        self.multiple_face_frames = 0
        self.eyes_closed_frames = 0
        self.too_close_frames = 0
        self.too_far_frames = 0
        
        # Thresholds
        self.LOOKING_AWAY_LIMIT = 60
        self.NO_FACE_LIMIT = 45
        self.MULTIPLE_FACE_LIMIT = 30
        self.EYES_CLOSED_LIMIT = 90
        self.DISTANCE_LIMIT = 45
        
        # Distance thresholds
        self.TOO_CLOSE_THRESHOLD = 0.40
        self.TOO_FAR_THRESHOLD = 0.15
        
        # Session tracking
        self.violations = []
        self.session_start = datetime.now()
        self.last_alert_time = 0
        self.alert_cooldown = 3
        
        # Stats
        self.total_frames = 0
        self.normal_frames = 0
        self.current_warnings = []
        
        # Gaze tracking
        self.gaze_history = []
        
    def play_alert(self, frequency=1000, duration=200):
        try:
            winsound.Beep(frequency, duration)
        except:
            pass
    
    def capture_violation(self, frame, violation_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"violations/{violation_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def add_violation(self, frame, violation_type, severity="MEDIUM"):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            screenshot = self.capture_violation(frame, violation_type)
            violation = {
                'time': datetime.now(),
                'type': violation_type,
                'severity': severity,
                'screenshot': screenshot
            }
            self.violations.append(violation)
            
            if severity == "HIGH":
                self.play_alert(1500, 300)
            elif severity == "MEDIUM":
                self.play_alert(1000, 200)
            else:
                self.play_alert(800, 150)
            
            self.last_alert_time = current_time
            print(f"⚠️  VIOLATION: {violation_type} [{severity}]")
    
    def detect_pupil(self, eye_region):
        """Enhanced pupil detection"""
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(eye_region, (7, 7), 0)
            
            # Threshold to find dark regions (pupil)
            _, threshold = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (likely the pupil)
                contour = max(contours, key=cv2.contourArea)
                
                # Calculate moments to find center
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy), cv2.contourArea(contour)
        except:
            pass
        
        return None, 0
    
    def analyze_gaze(self, eye_region, eye_width):
        """Analyze gaze direction from pupil position"""
        pupil_pos, area = self.detect_pupil(eye_region)
        
        if pupil_pos and area > 20:
            cx, cy = pupil_pos
            eye_center_x = eye_width // 2
            eye_center_y = eye_region.shape[0] // 2
            
            # Calculate deviation from center
            horizontal_deviation = abs(cx - eye_center_x) / eye_center_x
            vertical_deviation = abs(cy - eye_center_y) / eye_center_y
            
            # Determine direction
            direction = "CENTER"
            looking_away = False
            
            if horizontal_deviation > 0.35:
                looking_away = True
                if cx < eye_center_x:
                    direction = "LEFT"
                else:
                    direction = "RIGHT"
            
            if vertical_deviation > 0.40:
                looking_away = True
                if cy < eye_center_y:
                    direction = "UP"
                else:
                    direction = "DOWN"
            
            return pupil_pos, direction, looking_away, horizontal_deviation
        
        return None, "UNKNOWN", False, 0
    
    def analyze_frame(self, frame):
        self.total_frames += 1
        self.current_warnings = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        alert_status = "✓ MONITORING - NORMAL"
        alert_color = (0, 255, 0)
        violation_detected = False
        gaze_percentage = 0
        gaze_status = "UNKNOWN"
        
        h, w = frame.shape[:2]
        
        if len(faces) > 1:
            self.multiple_face_frames += 1
            self.current_warnings.append("⚠️ MULTIPLE PEOPLE DETECTED")
            if self.multiple_face_frames > self.MULTIPLE_FACE_LIMIT:
                alert_status = "🚨 VIOLATION: MULTIPLE PEOPLE"
                alert_color = (0, 0, 255)
                violation_detected = True
                self.add_violation(frame, "MULTIPLE_FACES", "HIGH")
                self.multiple_face_frames = 0
        elif len(faces) == 1:
            self.multiple_face_frames = 0
            self.no_face_frames = 0
            
            x, y, fw, fh = faces[0]
            
            # Calculate face size
            face_size = fh / h
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
            
            # Distance checks
            if face_size > self.TOO_CLOSE_THRESHOLD:
                self.too_close_frames += 1
                self.current_warnings.append("⚠️ TOO CLOSE TO CAMERA")
                if self.too_close_frames > self.DISTANCE_LIMIT:
                    alert_status = "🚨 VIOLATION: TOO CLOSE"
                    alert_color = (0, 165, 255)
                    violation_detected = True
                    self.add_violation(frame, "TOO_CLOSE", "MEDIUM")
                    self.too_close_frames = 0
            else:
                self.too_close_frames = max(0, self.too_close_frames - 2)
            
            if face_size < self.TOO_FAR_THRESHOLD:
                self.too_far_frames += 1
                self.current_warnings.append("⚠️ TOO FAR FROM CAMERA")
                if self.too_far_frames > self.DISTANCE_LIMIT:
                    alert_status = "🚨 VIOLATION: TOO FAR"
                    alert_color = (0, 165, 255)
                    violation_detected = True
                    self.add_violation(frame, "TOO_FAR", "MEDIUM")
                    self.too_far_frames = 0
            else:
                self.too_far_frames = max(0, self.too_far_frames - 2)
            
            # Detect eyes
            roi_gray = gray[y:y+fh, x:x+fw]
            roi_color = frame[y:y+fh, x:x+fw]
            
            # Focus on upper half of face for eyes
            upper_half_h = fh // 2
            roi_gray_upper = roi_gray[0:upper_half_h, :]
            roi_color_upper = roi_color[0:upper_half_h, :]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray_upper, 1.1, 5, minSize=(20, 20))
            
            if len(eyes) < 2:
                self.eyes_closed_frames += 1
                self.current_warnings.append("⚠️ EYES CLOSED OR NOT DETECTED")
                if self.eyes_closed_frames > self.EYES_CLOSED_LIMIT:
                    alert_status = "🚨 VIOLATION: EYES CLOSED/SLEEPING"
                    alert_color = (0, 165, 255)
                    violation_detected = True
                    self.add_violation(frame, "EYES_CLOSED", "MEDIUM")
                    self.eyes_closed_frames = 0
            else:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 3)
                
                # Analyze gaze for each eye
                gaze_directions = []
                looking_away_count = 0
                
                for (ex, ey, ew, eh) in eyes[:2]:
                    # Draw eye rectangle
                    cv2.rectangle(roi_color_upper, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Extract eye region
                    eye_region = roi_gray_upper[ey:ey+eh, ex:ex+ew]
                    
                    # Analyze pupil and gaze
                    pupil_pos, direction, is_away, deviation = self.analyze_gaze(eye_region, ew)
                    
                    if pupil_pos:
                        # Draw pupil on frame
                        pupil_x = x + ex + pupil_pos[0]
                        pupil_y = y + ey + pupil_pos[1]
                        cv2.circle(frame, (pupil_x, pupil_y), 4, (0, 255, 255), -1)
                        cv2.circle(frame, (pupil_x, pupil_y), 8, (0, 255, 0), 2)
                        
                        gaze_directions.append(direction)
                        if is_away:
                            looking_away_count += 1
                
                # Determine overall gaze
                if gaze_directions:
                    # Use most common direction
                    gaze_status = max(set(gaze_directions), key=gaze_directions.count)
                    
                    if looking_away_count >= 1:
                        self.looking_away_frames += 1
                        self.current_warnings.append(f"⚠️ LOOKING {gaze_status}")
                        gaze_percentage = min(100, int((self.looking_away_frames / self.LOOKING_AWAY_LIMIT) * 100))
                        
                        if self.looking_away_frames > self.LOOKING_AWAY_LIMIT:
                            alert_status = f"🚨 VIOLATION: LOOKING {gaze_status}"
                            alert_color = (0, 165, 255)
                            violation_detected = True
                            self.add_violation(frame, f"LOOKING_{gaze_status}", "MEDIUM")
                            self.looking_away_frames = 0
                    else:
                        self.looking_away_frames = max(0, self.looking_away_frames - 3)
                        gaze_percentage = min(100, int((self.looking_away_frames / self.LOOKING_AWAY_LIMIT) * 100))
        else:
            self.no_face_frames += 1
            self.current_warnings.append("⚠️ NO FACE DETECTED")
            if self.no_face_frames > self.NO_FACE_LIMIT:
                alert_status = "🚨 VIOLATION: STUDENT NOT IN FRAME"
                alert_color = (0, 0, 255)
                violation_detected = True
                self.add_violation(frame, "NO_FACE", "HIGH")
                self.no_face_frames = 0
        
        if not violation_detected and not self.current_warnings:
            self.normal_frames += 1
        
        return frame, alert_status, alert_color, gaze_percentage, gaze_status
    
    def draw_ui(self, frame, status, color, gaze_percentage, gaze_status):
        h, w = frame.shape[:2]
        
        # Top panel with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Main status
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Session info
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Session: {elapsed_str}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze Direction: {gaze_status}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Compliance score
        if self.total_frames > 0:
            compliance = int((self.normal_frames / self.total_frames) * 100)
            comp_color = (0, 255, 0) if compliance > 90 else (0, 165, 255) if compliance > 70 else (0, 0, 255)
            cv2.putText(frame, f"Compliance: {compliance}%", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, comp_color, 2)
        
        # Violation counter (top right)
        if len(self.violations) > 0:
            cv2.rectangle(frame, (w-200, 10), (w-10, 70), (0, 0, 200), -1)
            cv2.putText(frame, "VIOLATIONS", (w-185, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{len(self.violations)}", (w-110, 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        # Warning messages
        if self.current_warnings:
            warn_y = 200
            for warning in self.current_warnings[:3]:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (10, warn_y), (w-10, warn_y+35), (0, 0, 255), -1)
                cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, warning, (20, warn_y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                warn_y += 40
        
        # Gaze alert meter at bottom
        bar_y = h - 80
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (10, bar_y), (w-10, bar_y + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "GAZE ALERT METER", (20, bar_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        bar_width = w - 40
        bar_x = 20
        bar_y_pos = bar_y + 30
        
        # Draw meter outline
        cv2.rectangle(frame, (bar_x, bar_y_pos), (bar_x + bar_width, bar_y_pos + 20), (100, 100, 100), 2)
        
        # Fill meter based on gaze percentage
        if gaze_percentage > 0:
            fill_width = int((gaze_percentage / 100) * bar_width)
            if gaze_percentage < 50:
                bar_color = (0, 255, 255)  # Yellow
            elif gaze_percentage < 80:
                bar_color = (0, 165, 255)  # Orange
            else:
                bar_color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (bar_x, bar_y_pos), (bar_x + fill_width, bar_y_pos + 20), bar_color, -1)
            cv2.putText(frame, f"{gaze_percentage}%", (bar_x + bar_width + 10, bar_y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
        else:
            cv2.putText(frame, "0%", (bar_x + bar_width + 10, bar_y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def generate_report(self):
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PROFESSIONAL PROCTORING SESSION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Session Start: {self.session_start}\n")
            f.write(f"Session End: {datetime.now()}\n")
            f.write(f"Duration: {datetime.now() - self.session_start}\n\n")
            f.write(f"Total Frames Analyzed: {self.total_frames}\n")
            f.write(f"Normal Behavior: {self.normal_frames} frames ")
            f.write(f"({(self.normal_frames/max(1,self.total_frames)*100):.1f}%)\n\n")
            f.write(f"TOTAL VIOLATIONS: {len(self.violations)}\n")
            f.write("="*70 + "\n\n")
            
            if self.violations:
                f.write("VIOLATION DETAILS:\n")
                f.write("-"*70 + "\n")
                for i, v in enumerate(self.violations, 1):
                    f.write(f"\n{i}. {v['type']}\n")
                    f.write(f"   Time: {v['time']}\n")
                    f.write(f"   Severity: {v['severity']}\n")
                    f.write(f"   Screenshot: {v['screenshot']}\n")
            else:
                f.write("No violations detected. Excellent behavior!\n")
        
        print(f"\n📄 Report saved: {report_file}")
        return report_file
    
    def run(self):
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*70)
        print("PROFESSIONAL AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ Enhanced pupil detection and tracking")
        print("✓ Real-time gaze direction monitoring")
        print("✓ Distance monitoring (too close/too far)")
        print("✓ Real-time warning messages on screen")
        print("✓ Visual gaze alert bar")
        print("✓ Audio alerts on violations")
        print("\nPress ESC to end session and generate report")
        print("="*70 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            
            frame, status, color, gaze_pct, gaze_dir = self.analyze_frame(frame)
            frame = self.draw_ui(frame, status, color, gaze_pct, gaze_dir)
            
            cv2.imshow("Professional Proctoring System", frame)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
        
        webcam.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)
        self.generate_report()
        print(f"Total Violations: {len(self.violations)}")
        print("="*70 + "\n")

if __name__ == "__main__":
    proctor = ProfessionalProctoring()
    proctor.run()
