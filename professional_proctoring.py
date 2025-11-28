"""
PROFESSIONAL AI PROCTORING SYSTEM
Built on proven pupil detection from simple_gaze.py
Enhanced with professional proctoring features
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os

class ProfessionalProctoringSystem:
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
        
        # Thresholds
        self.LOOKING_AWAY_THRESHOLD = 90  # 3 seconds at 30fps
        self.NO_FACE_THRESHOLD = 60
        self.MULTIPLE_FACE_THRESHOLD = 45
        self.NO_EYES_THRESHOLD = 120
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_violation_time = {}
        self.violation_cooldown = 8
        self.total_frames = 0
        
        # Calibration
        self.calibration_frames = 0
        self.calibration_complete = False
        self.center_positions = []
        self.baseline_center = 0.5  # Default center
        
        # Gaze tracking
        self.current_gaze_status = "CENTER"
        
    def detect_pupil(self, eye_gray, ew, eh):
        """Simple and reliable pupil detection (from simple_gaze.py)"""
        try:
            # Threshold to find pupil (darkest point)
            _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (pupil)
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)
                
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return cx, cy
        except:
            pass
        
        return None, None
    
    def analyze_pupil_position(self, cx, ew):
        """Analyze if pupil is center, left, or right"""
        if cx is None:
            return "UNKNOWN", 0.5
        
        eye_center = ew // 2
        
        # Calculate position ratio (0 = far left, 0.5 = center, 1 = far right)
        ratio = cx / ew
        
        # During calibration
        if not self.calibration_complete:
            self.center_positions.append(ratio)
            if len(self.center_positions) >= 60:  # 2 seconds
                self.baseline_center = np.median(self.center_positions)
                self.calibration_complete = True
                print(f"✓ Calibration complete. Baseline: {self.baseline_center:.3f}")
            return "CALIBRATING", ratio
        
        # Adjust for baseline
        adjusted_ratio = ratio - self.baseline_center
        
        # Determine gaze direction with clear thresholds
        if adjusted_ratio < -0.15:
            return "RIGHT", ratio
        elif adjusted_ratio > 0.15:
            return "LEFT", ratio
        else:
            return "CENTER", ratio
    
    def log_violation(self, frame, violation_type, severity="MEDIUM"):
        """Log violation"""
        current_time = time.time()
        last_time = self.last_violation_time.get(violation_type, 0)
        
        if current_time - last_time > self.violation_cooldown:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"violations/{violation_type}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            violation = {
                'time': datetime.now(),
                'type': violation_type,
                'severity': severity,
                'screenshot': filename
            }
            self.violations.append(violation)
            self.last_violation_time[violation_type] = current_time
            
            # Beep
            try:
                import winsound
                winsound.Beep(1000, 300)
            except:
                pass
            
            print(f"⚠️  VIOLATION #{len(self.violations)}: {violation_type}")
    
    def process_frame(self, frame):
        """Main processing"""
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status = "✓ MONITORING"
        status_color = (0, 255, 0)
        gaze_text = ""
        violation_active = False
        eyes_detected_count = 0
        
        # Multiple faces check
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
        
        # No face check
        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count > self.NO_FACE_THRESHOLD:
                status = "⚠️ NO FACE DETECTED!"
                status_color = (0, 0, 255)
                violation_active = True
                self.log_violation(frame, "LEFT_FRAME", "HIGH")
                self.no_face_count = 0
            gaze_text = "No face detected"
        else:
            self.no_face_count = 0
        
        # Process face
        for (x, y, w, h) in faces:
            # Draw face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            eyes_detected_count = len(eyes)
            
            # Check eyes
            if len(eyes) < 2:
                self.no_eyes_count += 1
                if self.no_eyes_count > self.NO_EYES_THRESHOLD:
                    status = "⚠️ EYES NOT VISIBLE"
                    status_color = (0, 165, 255)
                    violation_active = True
                    self.log_violation(frame, "EYES_NOT_VISIBLE", "MEDIUM")
                    self.no_eyes_count = 0
                gaze_text = "Eyes not clearly visible"
            else:
                self.no_eyes_count = max(0, self.no_eyes_count - 2)
                
                # Analyze gaze from both eyes
                gaze_directions = []
                
                for (ex, ey, ew, eh) in eyes:
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Get eye region
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil
                    cx, cy = self.detect_pupil(eye_gray, ew, eh)
                    
                    if cx is not None and cy is not None:
                        # Analyze position
                        direction, ratio = self.analyze_pupil_position(cx, ew)
                        gaze_directions.append(direction)
                        
                        # Draw pupil
                        pupil_color = (0, 255, 0) if direction == "CENTER" else (0, 165, 255) if direction == "CALIBRATING" else (0, 0, 255)
                        cv2.circle(roi_color, (ex + cx, ey + cy), 3, pupil_color, -1)
                
                # Determine overall gaze
                if gaze_directions:
                    if "CALIBRATING" in gaze_directions:
                        gaze_text = "⏳ Calibrating... Look at screen"
                        self.current_gaze_status = "CALIBRATING"
                        status_color = (255, 165, 0)
                    else:
                        # Count directions
                        left_count = gaze_directions.count("LEFT")
                        right_count = gaze_directions.count("RIGHT")
                        center_count = gaze_directions.count("CENTER")
                        
                        if center_count >= 1:
                            gaze_text = "✓ Looking at screen"
                            self.current_gaze_status = "CENTER"
                            self.looking_away_count = max(0, self.looking_away_count - 2)
                        elif left_count > right_count:
                            gaze_text = "← Looking LEFT"
                            self.current_gaze_status = "LEFT"
                            self.looking_away_count += 1
                        elif right_count > left_count:
                            gaze_text = "→ Looking RIGHT"
                            self.current_gaze_status = "RIGHT"
                            self.looking_away_count += 1
                        else:
                            gaze_text = "~ Gaze detected"
                            self.current_gaze_status = "UNKNOWN"
                            self.looking_away_count += 0.5
                        
                        # Check threshold
                        if self.looking_away_count > self.LOOKING_AWAY_THRESHOLD:
                            status = "⚠️ LOOKING AWAY TOO LONG!"
                            status_color = (0, 165, 255)
                            violation_active = True
                            self.log_violation(frame, "LOOKING_AWAY", "MEDIUM")
                            self.looking_away_count = 0
                else:
                    gaze_text = "Pupils not detected"
        
        return frame, status, status_color, gaze_text, violation_active, eyes_detected_count
    
    def draw_ui(self, frame, status, status_color, gaze_text, violation_active, eyes_count):
        """Draw UI"""
        h, w = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        cv2.putText(frame, gaze_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        
        # Session info
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {elapsed_str}", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Top right info
        cv2.putText(frame, f"Violations: {len(self.violations)}", (w-250, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Eyes: {eyes_count}", (w-250, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Violation alert
        if len(self.violations) > 0:
            alert_color = (0, 0, 255) if violation_active else (0, 100, 200)
            cv2.rectangle(frame, (w-280, 85), (w-10, 120), alert_color, -1)
            cv2.putText(frame, f"⚠ WARNINGS: {len(self.violations)}", (w-270, 108), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom bar with gaze meter
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Gaze alert meter
        away_progress = min(100, int((self.looking_away_count / self.LOOKING_AWAY_THRESHOLD) * 100))
        
        # Background bar
        cv2.rectangle(frame, (20, h-60), (w-20, h-20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h-60), (w-20, h-20), (100, 100, 100), 2)
        
        # Progress bar
        if away_progress > 0:
            bar_width = int((w - 40) * (away_progress / 100))
            
            # Color based on level
            if away_progress < 30:
                bar_color = (0, 255, 0)
                level_text = "LOW"
            elif away_progress < 60:
                bar_color = (0, 220, 220)
                level_text = "MODERATE"
            elif away_progress < 85:
                bar_color = (0, 150, 255)
                level_text = "HIGH"
            else:
                bar_color = (0, 0, 255)
                level_text = "CRITICAL"
            
            cv2.rectangle(frame, (20, h-60), (20 + bar_width, h-20), bar_color, -1)
            cv2.putText(frame, f"GAZE ALERT: {away_progress}% - {level_text}", (30, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "GAZE ALERT: 0% - NORMAL", (30, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        # Instructions
        cv2.putText(frame, "Press ESC to end session", (w-280, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        return frame
    
    def generate_report(self):
        """Generate report"""
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PROFESSIONAL AI PROCTORING SYSTEM - SESSION REPORT\n")
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
                f.write("No violations detected. Excellent behavior!\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return report_file
    
    def run(self):
        """Main loop"""
        webcam = cv2.VideoCapture(0)
        
        if not webcam.isOpened():
            print("ERROR: Cannot access webcam!")
            return
        
        print("\n" + "="*70)
        print("PROFESSIONAL AI PROCTORING SYSTEM")
        print("="*70)
        print("✓ System initialized")
        print("✓ Look at the screen for 2 seconds to calibrate...")
        print("\nInstructions:")
        print("  - Sit 2-3 feet from camera")
        print("  - Ensure good lighting on your face")
        print("  - Look at screen during calibration")
        print("  - Press ESC to end session")
        print("="*70 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break
            
            # Process
            frame, status, color, gaze, violation, eyes = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, status, color, gaze, violation, eyes)
            
            # Show
            cv2.imshow("Professional AI Proctoring System", frame)
            
            # ESC to exit
            if cv2.waitKey(1) == 27:
                break
        
        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()
        
        # Report
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
    system = ProfessionalProctoringSystem()
    system.run()
