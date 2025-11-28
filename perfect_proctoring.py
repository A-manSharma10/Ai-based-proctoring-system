"""
PERFECT AI PROCTORING SYSTEM - Highly Refined and Accurate
- Better eye detection (filters out false positives)
- Smarter gaze tracking with calibration
- Adaptive thresholds
- Reduced false warnings
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os

class PerfectProctoringSystem:
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
        
        # REFINED Thresholds (more lenient)
        self.LOOKING_AWAY_THRESHOLD = 90  # 3 seconds
        self.NO_FACE_THRESHOLD = 60  # 2 seconds
        self.MULTIPLE_FACE_THRESHOLD = 45  # 1.5 seconds
        self.NO_EYES_THRESHOLD = 120  # 4 seconds
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_violation_time = {}  # Track per violation type
        self.violation_cooldown = 8  # seconds between same violation
        self.total_frames = 0
        
        # Calibration and smoothing
        self.gaze_history = []
        self.max_history = 15
        self.calibration_frames = 0
        self.baseline_deviation = 0
        
        # Face distance tracking
        self.face_size_history = []
        self.max_face_history = 30
        self.baseline_face_size = 0
        
    def is_valid_eye(self, ex, ey, ew, eh, face_w, face_h):
        """Filter out false eye detections (nose, mouth, etc.)"""
        # Eyes should be in upper 70% of face (more lenient)
        if ey > face_h * 0.7:
            return False
        
        # Eye aspect ratio check (eyes are wider than tall) - more lenient
        aspect_ratio = ew / eh
        if aspect_ratio < 0.8 or aspect_ratio > 4.0:
            return False
        
        # Eye size relative to face - more lenient
        eye_area = ew * eh
        face_area = face_w * face_h
        if eye_area < face_area * 0.005 or eye_area > face_area * 0.2:
            return False
        
        return True
    
    def get_valid_eyes(self, eyes, face_w, face_h):
        """Get only valid eye pairs"""
        valid_eyes = []
        
        for (ex, ey, ew, eh) in eyes:
            if self.is_valid_eye(ex, ey, ew, eh, face_w, face_h):
                valid_eyes.append((ex, ey, ew, eh))
        
        # If we have at least 2 valid eyes, return them
        if len(valid_eyes) >= 2:
            # Sort by y-coordinate and take top 2
            valid_eyes.sort(key=lambda e: e[1])
            
            eye1_y = valid_eyes[0][1]
            eye2_y = valid_eyes[1][1]
            
            # Eyes should be roughly at same height (more lenient)
            if abs(eye1_y - eye2_y) < face_h * 0.2:
                return valid_eyes[:2]
            else:
                # Still return them even if not perfectly aligned
                return valid_eyes[:2]
        
        # If filters are too strict, return best 2 eyes anyway
        if len(eyes) >= 2:
            eyes_list = list(eyes)
            eyes_list.sort(key=lambda e: e[1])  # Sort by y
            return eyes_list[:2]
        
        return []
    
    def detect_pupil_position(self, eye_gray, ew, eh):
        """Enhanced pupil detection with better accuracy"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(eye_gray, (3, 3), 0)
            
            # Simple threshold for pupil detection
            _, threshold = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (likely the pupil)
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                
                # Check if area is reasonable
                if area > 10:
                    M = cv2.moments(contour)
                    
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Validate pupil position
                        if 0 < cx < ew and 0 < cy < eh:
                            return cx, cy
        except:
            pass
        
        return None, None
    
    def calculate_gaze_deviation(self, cx, ew):
        """Calculate normalized gaze deviation"""
        eye_center = ew // 2
        deviation = (cx - eye_center) / eye_center
        return deviation
    
    def smooth_gaze(self, current_deviation):
        """Smooth gaze measurements to reduce jitter"""
        self.gaze_history.append(current_deviation)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        # Use median for robustness against outliers
        if len(self.gaze_history) >= 5:
            return np.median(self.gaze_history)
        return current_deviation
    
    def analyze_gaze(self, deviations):
        """Analyze gaze with adaptive thresholds"""
        if len(deviations) < 2:
            return "UNKNOWN", False, 0
        
        # Average deviation from both eyes
        avg_deviation = np.mean(deviations)
        
        # Smooth the measurement
        smoothed_deviation = self.smooth_gaze(avg_deviation)
        
        # Calibration phase (first 3 seconds)
        if self.calibration_frames < 90:
            self.calibration_frames += 1
            if self.calibration_frames == 90:
                self.baseline_deviation = np.mean(self.gaze_history) if self.gaze_history else 0
                print(f"✓ Calibration complete. Baseline: {self.baseline_deviation:.2f}")
            return "CALIBRATING", False, 0
        
        # Adjust for baseline
        adjusted_deviation = smoothed_deviation - self.baseline_deviation
        
        # More lenient thresholds
        if adjusted_deviation < -0.5:  # Looking right
            return "RIGHT", True, adjusted_deviation
        elif adjusted_deviation > 0.5:  # Looking left
            return "LEFT", True, adjusted_deviation
        elif abs(adjusted_deviation) < 0.35:  # Looking center
            return "CENTER", False, adjusted_deviation
        else:
            return "SLIGHT_AWAY", False, adjusted_deviation
    
    def update_face_baseline(self, face_w, face_h):
        """Track face size to detect if too close/far"""
        face_size = face_w * face_h
        self.face_size_history.append(face_size)
        
        if len(self.face_size_history) > self.max_face_history:
            self.face_size_history.pop(0)
        
        if len(self.face_size_history) >= 30 and self.baseline_face_size == 0:
            self.baseline_face_size = np.median(self.face_size_history)
    
    def check_face_distance(self, face_w, face_h):
        """Check if face is at appropriate distance"""
        if self.baseline_face_size == 0:
            return "CALIBRATING", True
        
        current_size = face_w * face_h
        ratio = current_size / self.baseline_face_size
        
        if ratio < 0.5:
            return "TOO_FAR", False
        elif ratio > 2.0:
            return "TOO_CLOSE", False
        else:
            return "GOOD", True
    
    def log_violation(self, frame, violation_type, severity="MEDIUM"):
        """Log violation with per-type cooldown"""
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
                winsound.Beep(800, 200)
            except:
                pass
            
            print(f"⚠️  VIOLATION #{len(self.violations)}: {violation_type}")
    
    def process_frame(self, frame):
        """Main processing with refined detection"""
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with stricter parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=6, 
            minSize=(150, 150),
            maxSize=(500, 500)
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
                status = "⚠️ MULTIPLE PEOPLE DETECTED!"
                status_color = (0, 0, 255)
                violation_active = True
                self.log_violation(frame, "MULTIPLE_FACES", "HIGH")
                self.multiple_face_count = 0
            detail_text = f"Multiple faces: {len(faces)}"
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
            detail_text = "Please position yourself in frame"
        else:
            self.no_face_count = 0
        
        # Process face
        for (x, y, w, h) in faces:
            # Update face distance baseline
            self.update_face_baseline(w, h)
            
            # Check face distance
            distance_status, distance_ok = self.check_face_distance(w, h)
            
            # Draw face rectangle
            face_color = (0, 255, 0) if distance_ok else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes with balanced parameters
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 15),
                maxSize=(100, 80)
            )
            
            # Filter and get valid eyes
            valid_eyes = self.get_valid_eyes(eyes, w, h)
            
            # Check if valid eyes detected
            if len(valid_eyes) < 2:
                self.no_eyes_count += 1
                if self.no_eyes_count > self.NO_EYES_THRESHOLD:
                    status = "⚠️ EYES NOT VISIBLE"
                    status_color = (0, 165, 255)
                    violation_active = True
                    self.log_violation(frame, "EYES_NOT_VISIBLE", "MEDIUM")
                    self.no_eyes_count = 0
                gaze_text = "Eyes not clearly visible"
                detail_text = "Adjust lighting or remove glasses"
            else:
                self.no_eyes_count = max(0, self.no_eyes_count - 2)
                
                # Analyze gaze
                deviations = []
                
                for (ex, ey, ew, eh) in valid_eyes:
                    # Draw eye rectangle
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Get eye region
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil
                    cx, cy = self.detect_pupil_position(eye_gray, ew, eh)
                    
                    if cx is not None:
                        deviation = self.calculate_gaze_deviation(cx, ew)
                        deviations.append(deviation)
                        
                        # Draw pupil
                        pupil_color = (0, 255, 0) if abs(deviation) < 0.4 else (0, 165, 255) if abs(deviation) < 0.6 else (0, 0, 255)
                        cv2.circle(roi_color, (ex + cx, ey + cy), 3, pupil_color, -1)
                
                # Analyze gaze direction
                if deviations:
                    direction, is_away, deviation_val = self.analyze_gaze(deviations)
                    
                    if direction == "CALIBRATING":
                        gaze_text = "⏳ Calibrating... Look at screen"
                        detail_text = f"Progress: {int((self.calibration_frames/90)*100)}%"
                        status_color = (255, 165, 0)
                    elif direction == "CENTER":
                        gaze_text = "✓ Looking at screen"
                        detail_text = f"Deviation: {abs(deviation_val):.2f}"
                        self.looking_away_count = max(0, self.looking_away_count - 3)
                    elif direction == "LEFT":
                        gaze_text = "← Looking LEFT"
                        detail_text = f"Deviation: {deviation_val:.2f}"
                        self.looking_away_count += 1
                    elif direction == "RIGHT":
                        gaze_text = "→ Looking RIGHT"
                        detail_text = f"Deviation: {deviation_val:.2f}"
                        self.looking_away_count += 1
                    else:
                        gaze_text = "~ Slight deviation"
                        detail_text = f"Deviation: {deviation_val:.2f}"
                        self.looking_away_count += 0.3
                    
                    # Check if looking away too long
                    if self.looking_away_count > self.LOOKING_AWAY_THRESHOLD:
                        status = "⚠️ LOOKING AWAY TOO LONG!"
                        status_color = (0, 165, 255)
                        violation_active = True
                        self.log_violation(frame, "LOOKING_AWAY", "MEDIUM")
                        self.looking_away_count = 0
            
            # Add distance warning
            if distance_status == "TOO_FAR":
                detail_text += " | Move closer to camera"
            elif distance_status == "TOO_CLOSE":
                detail_text += " | Move back from camera"
        
        return frame, status, status_color, gaze_text, detail_text, violation_active
    
    def draw_ui(self, frame, status, status_color, gaze_text, detail_text, violation_active):
        """Draw enhanced UI"""
        h, w = frame.shape[:2]
        
        # Top status bar (larger)
        cv2.rectangle(frame, (0, 0), (w, 130), (30, 30, 30), -1)
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2)
        cv2.putText(frame, gaze_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, detail_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Session info (top right)
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Time: {elapsed_str}", (w-250, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (w-250, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {self.total_frames}", (w-250, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Violation alert box
        if len(self.violations) > 0:
            alert_color = (0, 0, 255) if violation_active else (0, 100, 200)
            cv2.rectangle(frame, (w-280, 100), (w-10, 125), alert_color, -1)
            cv2.putText(frame, f"⚠ WARNINGS: {len(self.violations)}", (w-270, 118), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1)
        
        # Looking away progress bar
        away_progress = min(100, int((self.looking_away_count / self.LOOKING_AWAY_THRESHOLD) * 100))
        if away_progress > 0:
            bar_color = (0, 255, 0) if away_progress < 40 else (0, 200, 200) if away_progress < 70 else (0, 100, 255) if away_progress < 90 else (0, 0, 255)
            bar_width = int((w - 40) * (away_progress / 100))
            cv2.rectangle(frame, (20, h-45), (20 + bar_width, h-20), bar_color, -1)
            cv2.rectangle(frame, (20, h-45), (w-20, h-20), (100, 100, 100), 1)
            cv2.putText(frame, f"Gaze Alert: {away_progress}%", (25, h-28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press ESC to end | Sit 2-3 feet from camera", (w-400, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
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
                f.write("No violations detected. Excellent behavior!\n")
            
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
        
        # Set camera properties for better quality
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("PERFECT AI PROCTORING SYSTEM - ACTIVE")
        print("="*70)
        print("✓ System initialized")
        print("✓ Starting calibration (look at screen for 3 seconds)...")
        print("\nPress ESC to end session and generate report")
        print("="*70 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("ERROR: Cannot read frame from webcam")
                break
            
            # Process frame
            frame, status, color, gaze, detail, violation = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, status, color, gaze, detail, violation)
            
            # Show frame
            cv2.imshow("Perfect AI Proctoring System", frame)
            
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
    system = PerfectProctoringSystem()
    system.run()
