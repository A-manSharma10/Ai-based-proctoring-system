"""
ULTIMATE PROFESSIONAL AI PROCTORING SYSTEM
- Accurate pupil detection using dlib
- Real-time gaze tracking
- Distance detection (too close/too far)
- Real-time warning messages on screen
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
from collections import deque
from gaze_tracking import GazeTracking

class UltimateProctoring:
    def __init__(self):
        self.gaze = GazeTracking()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        os.makedirs('violations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        self.looking_away_frames = 0
        self.no_face_frames = 0
        self.multiple_face_frames = 0
        self.eyes_closed_frames = 0
        self.too_close_frames = 0
        self.too_far_frames = 0
        
        self.LOOKING_AWAY_LIMIT = 60
        self.NO_FACE_LIMIT = 45
        self.MULTIPLE_FACE_LIMIT = 30
        self.EYES_CLOSED_LIMIT = 90
        self.DISTANCE_LIMIT = 45
        
        self.TOO_CLOSE_THRESHOLD = 0.40
        self.TOO_FAR_THRESHOLD = 0.15
        
        self.violations = []
        self.session_start = datetime.now()
        self.last_alert_time = 0
        self.alert_cooldown = 3
        
        self.total_frames = 0
        self.normal_frames = 0
        self.current_warnings = []
        
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
    
    def calculate_face_size(self, face_rect, frame_shape):
        h, w = frame_shape[:2]
        x, y, fw, fh = face_rect
        face_size_ratio = fh / h
        return face_size_ratio
    
    def analyze_frame(self, frame):
        self.total_frames += 1
        self.current_warnings = []
        
        self.gaze.refresh(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        alert_status = "✓ MONITORING - NORMAL"
        alert_color = (0, 255, 0)
        violation_detected = False
        gaze_percentage = 0
        gaze_status = "UNKNOWN"
        
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
            
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_size = self.calculate_face_size(faces[0], frame.shape)
            
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
            
            if self.gaze.is_blinking():
                self.eyes_closed_frames += 1
                self.current_warnings.append("⚠️ EYES CLOSED")
                if self.eyes_closed_frames > self.EYES_CLOSED_LIMIT:
                    alert_status = "🚨 VIOLATION: EYES CLOSED/SLEEPING"
                    alert_color = (0, 165, 255)
                    violation_detected = True
                    self.add_violation(frame, "EYES_CLOSED", "MEDIUM")
                    self.eyes_closed_frames = 0
            else:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 3)
            
            if self.gaze.pupils_located and not self.gaze.is_blinking():
                left_pupil = self.gaze.pupil_left_coords()
                right_pupil = self.gaze.pupil_right_coords()
                
                if left_pupil and right_pupil:
                    cv2.circle(frame, left_pupil, 4, (0, 255, 255), -1)
                    cv2.circle(frame, right_pupil, 4, (0, 255, 255), -1)
                    cv2.circle(frame, left_pupil, 8, (0, 255, 0), 2)
                    cv2.circle(frame, right_pupil, 8, (0, 255, 0), 2)
                
                looking_away = False
                if self.gaze.is_right():
                    gaze_status = "RIGHT"
                    looking_away = True
                elif self.gaze.is_left():
                    gaze_status = "LEFT"
                    looking_away = True
                elif self.gaze.is_center():
                    gaze_status = "CENTER"
                    looking_away = False
                
                h_ratio = self.gaze.horizontal_ratio()
                v_ratio = self.gaze.vertical_ratio()
                
                if v_ratio is not None:
                    if v_ratio < 0.35:
                        gaze_status = "UP"
                        looking_away = True
                    elif v_ratio > 0.65:
                        gaze_status = "DOWN"
                        looking_away = True
                
                if looking_away:
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
                gaze_status = "DETECTING..."
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
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Session: {elapsed_str}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze Direction: {gaze_status}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.total_frames > 0:
            compliance = int((self.normal_frames / self.total_frames) * 100)
            comp_color = (0, 255, 0) if compliance > 90 else (0, 165, 255) if compliance > 70 else (0, 0, 255)
            cv2.putText(frame, f"Compliance: {compliance}%", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, comp_color, 2)
        
        if len(self.violations) > 0:
            cv2.rectangle(frame, (w-200, 10), (w-10, 70), (0, 0, 200), -1)
            cv2.putText(frame, "VIOLATIONS", (w-185, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{len(self.violations)}", (w-110, 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        if self.current_warnings:
            warn_y = 200
            for warning in self.current_warnings[:3]:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (10, warn_y), (w-10, warn_y+35), (0, 0, 255), -1)
                cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, warning, (20, warn_y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                warn_y += 40
        
        bar_y = h - 80
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (10, bar_y), (w-10, bar_y + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "GAZE ALERT METER", (20, bar_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        bar_width = w - 40
        bar_x = 20
        bar_y_pos = bar_y + 30
        
        cv2.rectangle(frame, (bar_x, bar_y_pos), (bar_x + bar_width, bar_y_pos + 20), (100, 100, 100), 2)
        
        if gaze_percentage > 0:
            fill_width = int((gaze_percentage / 100) * bar_width)
            if gaze_percentage < 50:
                bar_color = (0, 255, 255)
            elif gaze_percentage < 80:
                bar_color = (0, 165, 255)
            else:
                bar_color = (0, 0, 255)
            
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
            f.write("ULTIMATE PROFESSIONAL PROCTORING SESSION REPORT\n")
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
        print("ULTIMATE PROFESSIONAL AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ Accurate pupil detection and tracking")
        print("✓ Real-time gaze direction monitoring")
        print("✓ Distance monitoring (too close/too far)")
        print("✓ Real-time warning messages")
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
            
            cv2.imshow("Ultimate Professional Proctoring System", frame)
            
            if cv2.waitKey(1) == 27:
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
    proctor = UltimateProctoring()
    proctor.run()
