"""
ULTRA-REFINED AI PROCTORING SYSTEM
- Enhanced pupil detection that works at various distances
- Accurate gaze deviation tracking
- Proper calibration and thresholds
- Works perfectly at different face distances
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os

class UltraProctoringSystem:
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
        
        # Refined Thresholds
        self.LOOKING_AWAY_THRESHOLD = 100  # 3+ seconds
        self.NO_FACE_THRESHOLD = 60
        self.MULTIPLE_FACE_THRESHOLD = 45
        self.NO_EYES_THRESHOLD = 150
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_violation_time = {}
        self.violation_cooldown = 10
        self.total_frames = 0
        
        # Calibration
        self.calibration_frames = 0
        self.calibration_complete = False
        self.baseline_left = []
        self.baseline_right = []
        self.baseline_deviation = 0
        
        # Gaze smoothing
        self.gaze_history = []
        self.max_history = 10
        
    def enhance_eye_region(self, eye_gray):
        """Enhance eye image for better pupil detection"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(eye_gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return denoised
    
    def detect_pupil_advanced(self, eye_gray, ew, eh):
        """Advanced pupil detection that works at various distances"""
        try:
            # Enhance the eye region
            enhanced = self.enhance_eye_region(eye_gray)
            
            # Multiple threshold attempts for robustness
            thresholds = [25, 30, 35, 40]
            best_pupil = None
            best_score = 0
            
            for thresh_val in thresholds:
                # Apply threshold
                _, threshold = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Pupil should be reasonable size
                        min_area = (ew * eh) * 0.02
                        max_area = (ew * eh) * 0.5
                        
                        if min_area < area < max_area:
                            M = cv2.moments(contour)
                            if M['m00'] != 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                
                                # Check if position is valid
                                if 0 < cx < ew and 0 < cy < eh:
                                    # Score based on area and position
                                    # Prefer pupils closer to center
                                    center_dist = abs(cx - ew/2) + abs(cy - eh/2)
                                    score = area / (1 + center_dist)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_pupil = (cx, cy)
            
            return best_pupil
            
        except Exception as e:
            return None
    
    def calculate_gaze_ratio(self, pupil_x, eye_width):
        """Calculate gaze ratio (0 = right, 0.5 = center, 1 = left)"""
        if pupil_x is None or eye_width == 0:
            return None
        
        ratio = pupil_x / eye_width
        return ratio
    
    def analyze_gaze_direction(self, left_ratio, right_ratio):
        """Analyze gaze from both eyes with proper thresholds"""
        if left_ratio is None or right_ratio is None:
            return "UNKNOWN", 0, False
        
        # Average the ratios
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Smooth the measurement
        self.gaze_history.append(avg_ratio)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        smoothed_ratio = np.median(self.gaze_history) if len(self.gaze_history) >= 3 else avg_ratio
        
        # During calibration
        if not self.calibration_complete:
            if self.calibration_frames < 90:
                self.calibration_frames += 1
                return "CALIBRATING", smoothed_ratio, False
            else:
                # Calculate baseline
                if len(self.gaze_history) > 0:
                    self.baseline_deviation = np.median(self.gaze_history)
                self.calibration_complete = True
                print(f"✓ Calibration complete. Baseline: {self.baseline_deviation:.3f}")
                return "CALIBRATED", smoothed_ratio, False
        
        # Adjust for baseline
        adjusted_ratio = smoothed_ratio - self.baseline_deviation
        
        # Determine direction with proper thresholds
        # Center is around 0, left is positive, right is negative
        if adjusted_ratio < -0.15:  # Looking right
            return "RIGHT", adjusted_ratio, True
        elif adjusted_ratio > 0.15:  # Looking left
            return "LEFT", adjusted_ratio, True
        elif abs(adjusted_ratio) < 0.08:  # Looking center
            return "CENTER", adjusted_ratio, False
        else:  # Slight deviation
            return "SLIGHT", adjusted_ratio, False
    
    def log_violation(self, frame, violation_type, severity="MEDIUM"):
        """Log violation with cooldown"""
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
            
            # Play beep
            try:
                import winsound
                winsound.Beep(800, 250)
            except:
                pass
            
            print(f"⚠️  VIOLATION #{len(self.violations)}: {violation_type}")
    
    def process_frame(self, frame):
        """Main processing function"""
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.15, 
            minNeighbors=5, 
            minSize=(120, 120)
        )
        
        status = "✓ MONITORING"
        status_color = (0, 255, 0)
        gaze_text = ""
        detail_text = ""
        violation_active = False
        
        # Check multiple faces
        if len(faces) > 1:
            self.multiple_face_count += 1
            if self.multiple_face_count > self.MULTIPLE_FACE_THRESHOLD:
                status = "⚠️ MULTIPLE PEOPLE!"
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
            gaze_text = "No face detected"
            detail_text = "Position yourself in frame"
        else:
            self.no_face_count = 0
        
        # Process face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 15)
            )
            
            # Filter eyes (should be in upper part of face)
            valid_eyes = []
            for (ex, ey, ew, eh) in eyes:
                if ey < h * 0.65:  # Upper 65% of face
                    valid_eyes.append((ex, ey, ew, eh))
            
            # Sort by y-coordinate and take top 2
            valid_eyes.sort(key=lambda e: e[1])
            valid_eyes = valid_eyes[:2]
            
            # Check if eyes detected
            if len(valid_eyes) < 2:
                self.no_eyes_count += 1
                if self.no_eyes_count > self.NO_EYES_THRESHOLD:
                    status = "⚠️ EYES NOT VISIBLE"
                    status_color = (0, 165, 255)
                    violation_active = True
                    self.log_violation(frame, "EYES_NOT_VISIBLE", "MEDIUM")
                    self.no_eyes_count = 0
                gaze_text = "Eyes not detected"
                detail_text = "Improve lighting or adjust position"
            else:
                self.no_eyes_count = max(0, self.no_eyes_count - 2)
                
                # Analyze both eyes
                eye_ratios = []
                pupil_positions = []
                
                for i, (ex, ey, ew, eh) in enumerate(valid_eyes):
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Get eye region
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil
                    pupil = self.detect_pupil_advanced(eye_gray, ew, eh)
                    
                    if pupil:
                        px, py = pupil
                        pupil_positions.append((ex + px, ey + py))
                        
                        # Calculate gaze ratio
                        ratio = self.calculate_gaze_ratio(px, ew)
                        if ratio is not None:
                            eye_ratios.append(ratio)
                        
                        # Draw pupil
                        cv2.circle(roi_color, (ex + px, ey + py), 3, (0, 0, 255), -1)
                
                # Analyze gaze
                if len(eye_ratios) >= 2:
                    direction, deviation, is_away = self.analyze_gaze_direction(eye_ratios[0], eye_ratios[1])
                    
                    if direction == "CALIBRATING":
                        gaze_text = "⏳ Calibrating..."
                        detail_text = f"Look at screen: {int((self.calibration_frames/90)*100)}%"
                        status_color = (255, 165, 0)
                    elif direction == "CALIBRATED":
                        gaze_text = "✓ Calibration complete!"
                        detail_text = "System ready"
                        status_color = (0, 255, 0)
                    elif direction == "CENTER":
                        gaze_text = "✓ Looking at screen"
                        detail_text = f"Deviation: {abs(deviation):.3f}"
                        self.looking_away_count = max(0, self.looking_away_count - 3)
                    elif direction == "LEFT":
                        gaze_text = "← Looking LEFT"
                        detail_text = f"Deviation: {deviation:.3f}"
                        self.looking_away_count += 1.5
                    elif direction == "RIGHT":
                        gaze_text = "→ Looking RIGHT"
                        detail_text = f"Deviation: {deviation:.3f}"
                        self.looking_away_count += 1.5
                    else:  # SLIGHT
                        gaze_text = "~ Slight deviation"
                        detail_text = f"Deviation: {deviation:.3f}"
                        self.looking_away_count += 0.5
                    
                    # Check if looking away too long
                    if self.looking_away_count > self.LOOKING_AWAY_THRESHOLD:
                        status = "⚠️ LOOKING AWAY TOO LONG!"
                        status_color = (0, 165, 255)
                        violation_active = True
                        self.log_violation(frame, "LOOKING_AWAY", "MEDIUM")
                        self.looking_away_count = 0
                elif len(eye_ratios) == 1:
                    gaze_text = "One eye detected"
                    detail_text = "Turn face toward camera"
                else:
                    gaze_text = "Pupils not detected"
                    detail_text = "Adjust lighting or distance"
        
        return frame, status, status_color, gaze_text, detail_text, violation_active
    
    def draw_ui(self, frame, status, status_color, gaze_text, detail_text, violation_active):
        """Draw enhanced UI"""
        h, w = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 140), (25, 25, 25), -1)
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2)
        cv2.putText(frame, gaze_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
        cv2.putText(frame, detail_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        
        # Session info
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {elapsed_str}", (w-250, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (w-250, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Violation alert
        if len(self.violations) > 0:
            alert_color = (0, 0, 255) if violation_active else (0, 100, 200)
            cv2.rectangle(frame, (w-280, 75), (w-10, 105), alert_color, -1)
            cv2.putText(frame, f"⚠ WARNINGS: {len(self.violations)}", (w-270, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Bottom bar with gaze meter
        cv2.rectangle(frame, (0, h-70), (w, h), (25, 25, 25), -1)
        
        # Gaze away progress bar
        away_progress = min(100, int((self.looking_away_count / self.LOOKING_AWAY_THRESHOLD) * 100))
        
        # Draw background bar
        cv2.rectangle(frame, (20, h-55), (w-20, h-25), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h-55), (w-20, h-25), (100, 100, 100), 2)
        
        # Draw progress bar
        if away_progress > 0:
            bar_width = int((w - 40) * (away_progress / 100))
            
            # Color gradient based on progress
            if away_progress < 30:
                bar_color = (0, 255, 0)
            elif away_progress < 60:
                bar_color = (0, 220, 220)
            elif away_progress < 85:
                bar_color = (0, 150, 255)
            else:
                bar_color = (0, 0, 255)
            
            cv2.rectangle(frame, (20, h-55), (20 + bar_width, h-25), bar_color, -1)
            
            # Progress text
            cv2.putText(frame, f"Gaze Alert Level: {away_progress}%", (30, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Gaze Alert Level: 0%", (30, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "ESC to end | Sit 2-3 feet away", (w-350, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        
        return frame
    
    def generate_report(self):
        """Generate report"""
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ULTRA AI PROCTORING SYSTEM - SESSION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = datetime.now() - self.session_start
            f.write(f"Duration:      {str(duration).split('.')[0]}\n\n")
            
            f.write(f"Total Frames: {self.total_frames}\n")
            f.write(f"VIOLATIONS: {len(self.violations)}\n\n")
            
            if self.violations:
                f.write("="*70 + "\n")
                f.write("VIOLATION DETAILS\n")
                f.write("="*70 + "\n\n")
                
                violation_types = {}
                for v in self.violations:
                    violation_types[v['type']] = violation_types.get(v['type'], 0) + 1
                
                f.write("Summary:\n")
                for vtype, count in violation_types.items():
                    f.write(f"  - {vtype}: {count}\n")
                
                f.write("\n" + "-"*70 + "\n")
                f.write("Log:\n")
                f.write("-"*70 + "\n\n")
                
                for i, v in enumerate(self.violations, 1):
                    f.write(f"{i}. {v['type']}\n")
                    f.write(f"   Time: {v['time'].strftime('%H:%M:%S')}\n")
                    f.write(f"   Severity: {v['severity']}\n")
                    f.write(f"   Evidence: {v['screenshot']}\n\n")
            else:
                f.write("No violations detected!\n")
            
            f.write("="*70 + "\n")
        
        return report_file
    
    def run(self):
        """Main loop"""
        webcam = cv2.VideoCapture(0)
        
        if not webcam.isOpened():
            print("ERROR: Cannot access webcam!")
            return
        
        # Set camera properties
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("ULTRA AI PROCTORING SYSTEM - ACTIVE")
        print("="*70)
        print("✓ System initialized")
        print("✓ Look at the screen for 3 seconds to calibrate...")
        print("\nPress ESC to end session")
        print("="*70 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break
            
            # Process
            frame, status, color, gaze, detail, violation = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, status, color, gaze, detail, violation)
            
            # Show
            cv2.imshow("Ultra AI Proctoring System", frame)
            
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
        print(f"✓ Report: {report_file}")
        print(f"✓ Violations: {len(self.violations)}")
        if self.violations:
            print(f"✓ Evidence: violations/")
        print("="*70 + "\n")

if __name__ == "__main__":
    system = UltraProctoringSystem()
    system.run()
