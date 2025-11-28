"""
FINAL AI PROCTORING SYSTEM - Refined and Accurate
Combines working gaze tracking with proctoring features
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os

class FinalProctoringSystem:
    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create folders
        os.makedirs('violations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Violation tracking
        self.violations = []
        self.looking_away_count = 0
        self.no_face_count = 0
        self.multiple_face_count = 0
        self.no_eyes_count = 0
        
        # Thresholds (in frames)
        self.LOOKING_AWAY_THRESHOLD = 60  # 2 seconds at 30fps
        self.NO_FACE_THRESHOLD = 45  # 1.5 seconds
        self.MULTIPLE_FACE_THRESHOLD = 30  # 1 second
        self.NO_EYES_THRESHOLD = 90  # 3 seconds
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_violation_time = 0
        self.violation_cooldown = 5  # seconds between same violation type
        self.total_frames = 0
        
        # Gaze tracking
        self.gaze_history = []
        self.max_history = 10
        
    def log_violation(self, frame, violation_type, severity="MEDIUM"):
        """Log and save violation"""
        current_time = time.time()
        if current_time - self.last_violation_time > self.violation_cooldown:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations/{violation_type}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            violation = {
                'time': datetime.now(),
                'type': violation_type,
                'severity': severity,
                'screenshot': filename
            }
            self.violations.append(violation)
            self.last_violation_time = current_time
            
            # Play beep
            try:
                import winsound
                winsound.Beep(1000, 300)
            except:
                pass
            
            print(f"⚠️  VIOLATION #{len(self.violations)}: {violation_type}")
    
    def detect_gaze_direction(self, eye_gray, ew):
        """Accurate gaze detection"""
        try:
            # Threshold to find pupil
            _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (pupil)
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)
                
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    eye_center = ew // 2
                    
                    # Calculate deviation percentage
                    deviation = (cx - eye_center) / eye_center
                    
                    return cx, deviation
        except:
            pass
        
        return None, 0
    
    def analyze_gaze(self, deviations):
        """Analyze gaze from both eyes"""
        if len(deviations) < 2:
            return "UNKNOWN", False
        
        avg_deviation = np.mean(deviations)
        
        # Thresholds for looking away
        if avg_deviation < -0.4:  # Looking right
            return "RIGHT", True
        elif avg_deviation > 0.4:  # Looking left
            return "LEFT", True
        elif abs(avg_deviation) < 0.25:  # Looking center
            return "CENTER", False
        else:
            return "SLIGHT_AWAY", False
    
    def process_frame(self, frame):
        """Main processing function"""
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))
        
        status = "✓ MONITORING"
        status_color = (0, 255, 0)
        gaze_text = ""
        violation_active = False
        
        # Check multiple faces
        if len(faces) > 1:
            self.multiple_face_count += 1
            if self.multiple_face_count > self.MULTIPLE_FACE_THRESHOLD:
                status = "⚠️ MULTIPLE PEOPLE DETECTED!"
                status_color = (0, 0, 255)
                violation_active = True
                self.log_violation(frame, "MULTIPLE_FACES", "HIGH")
                self.multiple_face_count = 0
        else:
            self.multiple_face_count = 0
        
        # Check no face
        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count > self.NO_FACE_THRESHOLD:
                status = "⚠️ NO FACE DETECTED!"
                status_color = (0, 0, 255)
                violation_active = True
                self.log_violation(frame, "LEFT_FRAME", "HIGH")
                self.no_face_count = 0
            gaze_text = "No face in frame"
        else:
            self.no_face_count = 0
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
            
            # Check if eyes detected
            if len(eyes) < 2:
                self.no_eyes_count += 1
                if self.no_eyes_count > self.NO_EYES_THRESHOLD:
                    status = "⚠️ EYES NOT VISIBLE / CLOSED"
                    status_color = (0, 165, 255)
                    violation_active = True
                    self.log_violation(frame, "EYES_CLOSED", "MEDIUM")
                    self.no_eyes_count = 0
                gaze_text = "Eyes not detected"
            else:
                self.no_eyes_count = max(0, self.no_eyes_count - 1)
                
                # Analyze gaze for each eye
                deviations = []
                pupil_positions = []
                
                for (ex, ey, ew, eh) in eyes[:2]:  # Only first 2 eyes
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Get eye region
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil position
                    cx, deviation = self.detect_gaze_direction(eye_gray, ew)
                    
                    if cx is not None:
                        deviations.append(deviation)
                        pupil_positions.append((ex + cx, ey + eh//2))
                        
                        # Draw pupil
                        if abs(deviation) > 0.35:
                            cv2.circle(roi_color, (ex + cx, ey + eh//2), 4, (0, 0, 255), -1)
                        else:
                            cv2.circle(roi_color, (ex + cx, ey + eh//2), 4, (0, 255, 0), -1)
                
                # Analyze gaze direction
                if deviations:
                    direction, is_away = self.analyze_gaze(deviations)
                    
                    if direction == "CENTER":
                        gaze_text = "✓ Looking at screen"
                        self.looking_away_count = max(0, self.looking_away_count - 2)
                    elif direction == "LEFT":
                        gaze_text = "← Looking LEFT"
                        self.looking_away_count += 1
                    elif direction == "RIGHT":
                        gaze_text = "→ Looking RIGHT"
                        self.looking_away_count += 1
                    else:
                        gaze_text = "~ Slight deviation"
                        self.looking_away_count += 0.5
                    
                    # Check if looking away too long
                    if self.looking_away_count > self.LOOKING_AWAY_THRESHOLD:
                        status = "⚠️ LOOKING AWAY FROM SCREEN!"
                        status_color = (0, 165, 255)
                        violation_active = True
                        self.log_violation(frame, "LOOKING_AWAY", "MEDIUM")
                        self.looking_away_count = 0
        
        return frame, status, status_color, gaze_text, violation_active
    
    def draw_ui(self, frame, status, status_color, gaze_text, violation_active):
        """Draw professional UI"""
        h, w = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        cv2.putText(frame, gaze_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Session info (top right)
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {elapsed_str}", (w-250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (w-250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Violation alert box
        if len(self.violations) > 0:
            alert_color = (0, 0, 255) if violation_active else (0, 100, 200)
            cv2.rectangle(frame, (w-280, 70), (w-10, 95), alert_color, -1)
            cv2.putText(frame, f"⚠ TOTAL WARNINGS: {len(self.violations)}", (w-270, 88), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, h-50), (w, h), (40, 40, 40), -1)
        
        # Looking away progress bar
        away_progress = min(100, int((self.looking_away_count / self.LOOKING_AWAY_THRESHOLD) * 100))
        if away_progress > 0:
            bar_color = (0, 255, 0) if away_progress < 50 else (0, 165, 255) if away_progress < 80 else (0, 0, 255)
            bar_width = int((w - 40) * (away_progress / 100))
            cv2.rectangle(frame, (20, h-35), (20 + bar_width, h-15), bar_color, -1)
            cv2.putText(frame, f"Gaze Alert: {away_progress}%", (25, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press ESC to end session", (w-250, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def generate_report(self):
        """Generate detailed report"""
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("AI PROCTORING SYSTEM - SESSION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = datetime.now() - self.session_start
            f.write(f"Duration:      {str(duration).split('.')[0]}\n\n")
            
            f.write(f"Total Frames Analyzed: {self.total_frames}\n")
            f.write(f"TOTAL VIOLATIONS: {len(self.violations)}\n\n")
            
            if self.violations:
                f.write("="*70 + "\n")
                f.write("VIOLATION DETAILS\n")
                f.write("="*70 + "\n\n")
                
                violation_types = {}
                for v in self.violations:
                    violation_types[v['type']] = violation_types.get(v['type'], 0) + 1
                
                f.write("Summary by Type:\n")
                for vtype, count in violation_types.items():
                    f.write(f"  - {vtype}: {count}\n")
                
                f.write("\n" + "-"*70 + "\n")
                f.write("Chronological Log:\n")
                f.write("-"*70 + "\n\n")
                
                for i, v in enumerate(self.violations, 1):
                    f.write(f"{i}. {v['type']}\n")
                    f.write(f"   Time: {v['time'].strftime('%H:%M:%S')}\n")
                    f.write(f"   Severity: {v['severity']}\n")
                    f.write(f"   Evidence: {v['screenshot']}\n\n")
            else:
                f.write("✓ No violations detected. Excellent behavior!\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_file
    
    def run(self):
        """Main execution loop"""
        webcam = cv2.VideoCapture(0)
        
        if not webcam.isOpened():
            print("ERROR: Cannot access webcam!")
            return
        
        print("\n" + "="*70)
        print("AI PROCTORING SYSTEM - ACTIVE")
        print("="*70)
        print("✓ System initialized successfully")
        print("✓ Monitoring started...")
        print("\nPress ESC to end session and generate report")
        print("="*70 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("ERROR: Cannot read frame from webcam")
                break
            
            # Process frame
            frame, status, color, gaze, violation = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, status, color, gaze, violation)
            
            # Show frame
            cv2.imshow("AI Proctoring System", frame)
            
            # Check for ESC key
            if cv2.waitKey(1) == 27:
                break
        
        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()
        
        # Generate report
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)
        report_file = self.generate_report()
        print(f"✓ Report saved: {report_file}")
        print(f"✓ Total violations: {len(self.violations)}")
        if self.violations:
            print(f"✓ Evidence saved in: violations/")
        print("="*70 + "\n")

if __name__ == "__main__":
    system = FinalProctoringSystem()
    system.run()
