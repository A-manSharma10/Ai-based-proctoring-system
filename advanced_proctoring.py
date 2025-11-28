"""
Advanced AI Proctoring System with Audio Alerts
Features:
- Real-time face and eye tracking
- Audio warnings
- Suspicious behavior detection
- Session report generation
- Screenshot capture on violations
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os
import winsound

class AdvancedProctoring:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create folders
        os.makedirs('violations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Tracking
        self.violations = []
        self.looking_away_frames = 0
        self.no_face_frames = 0
        self.multiple_face_frames = 0
        self.eyes_closed_frames = 0
        
        # Thresholds
        self.LOOKING_AWAY_LIMIT = 45  # ~1.5 seconds at 30fps
        self.NO_FACE_LIMIT = 30
        self.MULTIPLE_FACE_LIMIT = 15
        self.EYES_CLOSED_LIMIT = 60
        
        # Session info
        self.session_start = datetime.now()
        self.last_alert_time = 0
        self.alert_cooldown = 5
        
        # Stats
        self.total_frames = 0
        self.normal_frames = 0
        
    def play_alert(self):
        """Play warning sound"""
        try:
            winsound.Beep(1000, 200)  # 1000Hz for 200ms
        except:
            pass
    
    def capture_violation(self, frame, violation_type):
        """Save screenshot of violation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violations/{violation_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def add_violation(self, frame, violation_type, severity="MEDIUM"):
        """Log violation with details"""
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
            self.play_alert()
            self.last_alert_time = current_time
            print(f"\n⚠️  VIOLATION DETECTED: {violation_type}")
    
    def detect_eyes_closed(self, eyes_detected, face_detected):
        """Check if eyes are closed"""
        if face_detected and eyes_detected < 2:
            return True
        return False
    
    def analyze_frame(self, frame):
        """Main analysis function"""
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        alert_status = "MONITORING"
        alert_color = (0, 255, 0)
        violation_detected = False
        
        # Multiple faces check
        if len(faces) > 1:
            self.multiple_face_frames += 1
            if self.multiple_face_frames > self.MULTIPLE_FACE_LIMIT:
                alert_status = "⚠️ MULTIPLE PEOPLE DETECTED"
                alert_color = (0, 0, 255)
                violation_detected = True
                self.add_violation(frame, "MULTIPLE_FACES", "HIGH")
                self.multiple_face_frames = 0
        else:
            self.multiple_face_frames = 0
        
        # No face check
        if len(faces) == 0:
            self.no_face_frames += 1
            if self.no_face_frames > self.NO_FACE_LIMIT:
                alert_status = "⚠️ STUDENT NOT IN FRAME"
                alert_color = (0, 0, 255)
                violation_detected = True
                self.add_violation(frame, "NO_FACE", "HIGH")
                self.no_face_frames = 0
        else:
            self.no_face_frames = 0
        
        # Process face
        eyes_detected = 0
        gaze_away = False
        
        for (x, y, w, h) in faces:
            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            eyes_detected = len(eyes)
            
            # Check eyes closed
            if self.detect_eyes_closed(eyes_detected, True):
                self.eyes_closed_frames += 1
                if self.eyes_closed_frames > self.EYES_CLOSED_LIMIT:
                    alert_status = "⚠️ EYES CLOSED / SLEEPING"
                    alert_color = (0, 165, 255)
                    violation_detected = True
                    self.add_violation(frame, "EYES_CLOSED", "MEDIUM")
                    self.eyes_closed_frames = 0
            else:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 1)
            
            # Analyze gaze
            away_count = 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Detect pupil
                _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        eye_center = ew // 2
                        deviation = abs(cx - eye_center)
                        
                        if deviation > ew * 0.35:
                            away_count += 1
                            cv2.circle(roi_color, (ex + cx, ey + eh//2), 3, (0, 0, 255), -1)
                        else:
                            cv2.circle(roi_color, (ex + cx, ey + eh//2), 3, (0, 255, 0), -1)
            
            if away_count >= 2:
                gaze_away = True
        
        # Check looking away
        if gaze_away:
            self.looking_away_frames += 1
            if self.looking_away_frames > self.LOOKING_AWAY_LIMIT:
                alert_status = "⚠️ LOOKING AWAY FROM SCREEN"
                alert_color = (0, 165, 255)
                violation_detected = True
                self.add_violation(frame, "LOOKING_AWAY", "MEDIUM")
                self.looking_away_frames = 0
        else:
            self.looking_away_frames = max(0, self.looking_away_frames - 2)
        
        if not violation_detected:
            self.normal_frames += 1
        
        return frame, alert_status, alert_color, eyes_detected
    
    def draw_ui(self, frame, status, color, eyes_detected):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Top panel
        cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        
        # Session info
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Session Time: {elapsed_str}", (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Eyes Detected: {eyes_detected}", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Violation counter (top right)
        if len(self.violations) > 0:
            cv2.rectangle(frame, (w-250, 10), (w-10, 70), (0, 0, 255), -1)
            cv2.putText(frame, f"VIOLATIONS: {len(self.violations)}", (w-240, 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bars
        bar_y = h - 60
        cv2.rectangle(frame, (10, bar_y), (310, bar_y + 40), (0, 0, 0), -1)
        
        # Looking away indicator
        away_pct = min(100, int((self.looking_away_frames / self.LOOKING_AWAY_LIMIT) * 100))
        if away_pct > 0:
            bar_color = (0, 255, 255) if away_pct < 70 else (0, 165, 255)
            cv2.rectangle(frame, (15, bar_y + 5), (15 + int(away_pct * 2.9), bar_y + 20), bar_color, -1)
            cv2.putText(frame, f"Gaze: {away_pct}%", (15, bar_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def generate_report(self):
        """Generate session report"""
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PROCTORING SESSION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Session Start: {self.session_start}\n")
            f.write(f"Session End: {datetime.now()}\n")
            f.write(f"Duration: {datetime.now() - self.session_start}\n\n")
            f.write(f"Total Frames Analyzed: {self.total_frames}\n")
            f.write(f"Normal Behavior: {self.normal_frames} frames ({(self.normal_frames/max(1,self.total_frames)*100):.1f}%)\n\n")
            f.write(f"TOTAL VIOLATIONS: {len(self.violations)}\n")
            f.write("="*60 + "\n\n")
            
            if self.violations:
                f.write("VIOLATION DETAILS:\n")
                f.write("-"*60 + "\n")
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
        """Main loop"""
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        print("\n" + "="*60)
        print("ADVANCED AI PROCTORING SYSTEM")
        print("="*60)
        print("System is now monitoring...")
        print("Press ESC to end session and generate report")
        print("="*60 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            
            frame, status, color, eyes = self.analyze_frame(frame)
            frame = self.draw_ui(frame, status, color, eyes)
            
            cv2.imshow("Advanced Proctoring System", frame)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
        
        webcam.release()
        cv2.destroyAllWindows()
        
        # Generate report
        print("\n" + "="*60)
        print("SESSION ENDED")
        print("="*60)
        self.generate_report()
        print(f"Total Violations: {len(self.violations)}")
        print("="*60 + "\n")

if __name__ == "__main__":
    proctor = AdvancedProctoring()
    proctor.run()
