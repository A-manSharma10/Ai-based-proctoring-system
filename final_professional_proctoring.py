"""
FINAL PROFESSIONAL AI PROCTORING SYSTEM
- Filters out nose, mouth, ears (only detects real eyes)
- Accurate pupil movement tracking
- Proper gaze detection at all distances
- Professional-grade proctoring
"""
import cv2
import numpy as np
from datetime import datetime
import time
import os

class FinalProfessionalProctoring:
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
        self.LOOKING_AWAY_THRESHOLD = 90
        self.NO_FACE_THRESHOLD = 60
        self.MULTIPLE_FACE_THRESHOLD = 45
        self.NO_EYES_THRESHOLD = 120
        
        # Session
        self.session_start = datetime.now()
        self.last_violation_time = {}
        self.violation_cooldown = 8
        self.total_frames = 0
        
        # Calibration
        self.calibration_frames = 0
        self.calibration_complete = False
        self.baseline_positions = []
        self.baseline_left = 0.5
        self.baseline_right = 0.5
        
    def filter_real_eyes(self, eyes, face_w, face_h):
        """Filter to get only real eyes (not nose, mouth, ears)"""
        valid_eyes = []
        
        for (ex, ey, ew, eh) in eyes:
            # 1. Eyes must be in upper 60% of face
            if ey > face_h * 0.6:
                continue
            
            # 2. Eyes must not be at very edges (filters ears)
            if ex < face_w * 0.05 or ex > face_w * 0.95:
                continue
            
            # 3. Eye aspect ratio (width/height should be reasonable)
            aspect_ratio = ew / eh
            if aspect_ratio < 0.8 or aspect_ratio > 3.5:
                continue
            
            # 4. Eye size should be reasonable relative to face
            eye_area = ew * eh
            face_area = face_w * face_h
            area_ratio = eye_area / face_area
            if area_ratio < 0.01 or area_ratio > 0.15:
                continue
            
            # 5. Eyes should be in reasonable horizontal position
            eye_center_x = ex + ew // 2
            if eye_center_x < face_w * 0.15 or eye_center_x > face_w * 0.85:
                continue
            
            valid_eyes.append((ex, ey, ew, eh))
        
        # Get the two eyes that are most likely real (top two, horizontally separated)
        if len(valid_eyes) >= 2:
            # Sort by y-coordinate (top to bottom)
            valid_eyes.sort(key=lambda e: e[1])
            
            # Take top candidates
            top_eyes = valid_eyes[:4]
            
            # Find the pair with best horizontal separation
            best_pair = None
            best_separation = 0
            
            for i in range(len(top_eyes)):
                for j in range(i + 1, len(top_eyes)):
                    eye1 = top_eyes[i]
                    eye2 = top_eyes[j]
                    
                    # Check if they're at similar height (real eyes are aligned)
                    y_diff = abs(eye1[1] - eye2[1])
                    if y_diff < face_h * 0.15:
                        # Check horizontal separation
                        x_separation = abs((eye1[0] + eye1[2]//2) - (eye2[0] + eye2[2]//2))
                        
                        if x_separation > best_separation:
                            best_separation = x_separation
                            best_pair = [eye1, eye2]
            
            if best_pair:
                # Sort left to right
                best_pair.sort(key=lambda e: e[0])
                return best_pair
        
        return []
    
    def detect_pupil(self, eye_gray, ew, eh):
        """Reliable pupil detection"""
        try:
            # Threshold to find darkest regions (pupil)
            _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the best pupil candidate
                best_contour = None
                best_score = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Pupil should be reasonable size
                    if area < 10 or area > (ew * eh * 0.6):
                        continue
                    
                    # Get center
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Pupil should be within eye bounds
                        if 0 < cx < ew and 0 < cy < eh:
                            # Score based on area and centrality
                            center_dist = abs(cx - ew/2) + abs(cy - eh/2)
                            score = area / (1 + center_dist * 0.1)
                            
                            if score > best_score:
                                best_score = score
                                best_contour = (cx, cy)
                
                if best_contour:
                    return best_contour
        except:
            pass
        
        return None, None
    
    def calculate_gaze_position(self, pupil_x, eye_width):
        """Calculate normalized gaze position (0 to 1)"""
        if pupil_x is None or eye_width == 0:
            return None
        
        # Normalize to 0-1 range
        position = pupil_x / eye_width
        return position
    
    def analyze_gaze(self, left_pos, right_pos):
        """Analyze gaze direction from both eyes"""
        if left_pos is None or right_pos is None:
            return "UNKNOWN", 0.5, False
        
        # Average position from both eyes
        avg_position = (left_pos + right_pos) / 2
        
        # During calibration
        if not self.calibration_complete:
            self.baseline_positions.append(avg_position)
            self.calibration_frames += 1
            
            if self.calibration_frames >= 60:  # 2 seconds
                self.baseline_left = np.percentile(self.baseline_positions, 50)
                self.baseline_right = self.baseline_left
                self.calibration_complete = True
                print(f"✓ Calibration complete. Baseline: {self.baseline_left:.3f}")
                return "CALIBRATED", avg_position, False
            
            return "CALIBRATING", avg_position, False
        
        # Calculate deviation from baseline
        deviation = avg_position - self.baseline_left
        
        # Determine gaze direction
        if deviation < -0.12:  # Looking right (pupil moved left in eye)
            return "RIGHT", avg_position, True
        elif deviation > 0.12:  # Looking left (pupil moved right in eye)
            return "LEFT", avg_position, True
        elif abs(deviation) < 0.08:  # Looking center
            return "CENTER", avg_position, False
        else:
            return "SLIGHT", avg_position, False
    
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
        detail_text = ""
        violation_active = False
        eyes_detected = 0
        
        # Multiple faces
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
        
        # No face
        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count > self.NO_FACE_THRESHOLD:
                status = "⚠️ NO FACE!"
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
            # Draw face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect all eyes
            all_eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            # Filter to get real eyes only
            real_eyes = self.filter_real_eyes(all_eyes, w, h)
            eyes_detected = len(real_eyes)
            
            # Check eyes
            if len(real_eyes) < 2:
                self.no_eyes_count += 1
                if self.no_eyes_count > self.NO_EYES_THRESHOLD:
                    status = "⚠️ EYES NOT VISIBLE"
                    status_color = (0, 165, 255)
                    violation_active = True
                    self.log_violation(frame, "EYES_NOT_VISIBLE", "MEDIUM")
                    self.no_eyes_count = 0
                gaze_text = "Eyes not detected"
                detail_text = "Adjust lighting or face camera"
            else:
                self.no_eyes_count = max(0, self.no_eyes_count - 2)
                
                # Process both eyes
                left_eye = real_eyes[0]  # Left eye (leftmost)
                right_eye = real_eyes[1]  # Right eye (rightmost)
                
                pupil_positions = []
                
                for i, (ex, ey, ew, eh) in enumerate([left_eye, right_eye]):
                    # Draw eye
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Get eye region
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    # Detect pupil
                    px, py = self.detect_pupil(eye_gray, ew, eh)
                    
                    if px is not None:
                        # Calculate position
                        position = self.calculate_gaze_position(px, ew)
                        pupil_positions.append(position)
                        
                        # Draw pupil
                        cv2.circle(roi_color, (ex + px, ey + py), 4, (0, 0, 255), -1)
                        cv2.circle(roi_color, (ex + px, ey + py), 5, (255, 255, 255), 1)
                
                # Analyze gaze
                if len(pupil_positions) == 2:
                    direction, position, is_away = self.analyze_gaze(pupil_positions[0], pupil_positions[1])
                    
                    if direction == "CALIBRATING":
                        gaze_text = "⏳ Calibrating..."
                        detail_text = f"Progress: {int((self.calibration_frames/60)*100)}%"
                        status_color = (255, 165, 0)
                    elif direction == "CALIBRATED":
                        gaze_text = "✓ Calibration complete!"
                        detail_text = "System ready"
                    elif direction == "CENTER":
                        gaze_text = "✓ Looking at screen"
                        detail_text = f"Position: {position:.3f}"
                        self.looking_away_count = max(0, self.looking_away_count - 2)
                    elif direction == "LEFT":
                        gaze_text = "← Looking LEFT"
                        detail_text = f"Position: {position:.3f}"
                        self.looking_away_count += 1
                    elif direction == "RIGHT":
                        gaze_text = "→ Looking RIGHT"
                        detail_text = f"Position: {position:.3f}"
                        self.looking_away_count += 1
                    else:  # SLIGHT
                        gaze_text = "~ Slight deviation"
                        detail_text = f"Position: {position:.3f}"
                        self.looking_away_count += 0.3
                    
                    # Check threshold
                    if self.looking_away_count > self.LOOKING_AWAY_THRESHOLD:
                        status = "⚠️ LOOKING AWAY TOO LONG!"
                        status_color = (0, 165, 255)
                        violation_active = True
                        self.log_violation(frame, "LOOKING_AWAY", "MEDIUM")
                        self.looking_away_count = 0
                else:
                    gaze_text = "Pupils not fully detected"
                    detail_text = "Ensure good lighting"
        
        return frame, status, status_color, gaze_text, detail_text, violation_active, eyes_detected
    
    def draw_ui(self, frame, status, status_color, gaze_text, detail_text, violation_active, eyes_count):
        """Draw professional UI"""
        h, w = frame.shape[:2]
        
        # Top bar with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Status
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        cv2.putText(frame, gaze_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)
        cv2.putText(frame, detail_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 1)
        
        # Session info (top right)
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Session: {elapsed_str}", (w-280, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (w-280, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"Eyes: {eyes_count}/2", (w-280, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Violation alert box
        if len(self.violations) > 0:
            alert_color = (0, 0, 255) if violation_active else (0, 100, 200)
            cv2.rectangle(frame, (w-300, 115), (w-10, 155), alert_color, -1)
            cv2.putText(frame, f"⚠ TOTAL WARNINGS: {len(self.violations)}", (w-290, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom bar with gaze meter
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Gaze alert meter
        away_progress = min(100, int((self.looking_away_count / self.LOOKING_AWAY_THRESHOLD) * 100))
        
        # Background bar
        cv2.rectangle(frame, (20, h-70), (w-20, h-25), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, h-70), (w-20, h-25), (120, 120, 120), 2)
        
        # Progress bar with gradient
        if away_progress > 0:
            bar_width = int((w - 40) * (away_progress / 100))
            
            # Color gradient
            if away_progress < 25:
                bar_color = (0, 255, 0)
                level = "NORMAL"
            elif away_progress < 50:
                bar_color = (0, 230, 230)
                level = "LOW"
            elif away_progress < 75:
                bar_color = (0, 180, 255)
                level = "MODERATE"
            elif away_progress < 90:
                bar_color = (0, 100, 255)
                level = "HIGH"
            else:
                bar_color = (0, 0, 255)
                level = "CRITICAL"
            
            cv2.rectangle(frame, (20, h-70), (20 + bar_width, h-25), bar_color, -1)
            cv2.putText(frame, f"GAZE ALERT: {away_progress}% - {level}", (35, h-43), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "GAZE ALERT: 0% - NORMAL", (35, h-43), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1)
        
        # Instructions
        cv2.putText(frame, "Press ESC to end session and generate report", (w-420, h-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
        
        return frame
    
    def generate_report(self):
        """Generate detailed report"""
        report_file = f"reports/session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*75 + "\n")
            f.write("PROFESSIONAL AI PROCTORING SYSTEM - SESSION REPORT\n")
            f.write("="*75 + "\n\n")
            
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = datetime.now() - self.session_start
            f.write(f"Duration:      {str(duration).split('.')[0]}\n\n")
            
            f.write(f"Total Frames Analyzed: {self.total_frames}\n")
            f.write(f"TOTAL VIOLATIONS: {len(self.violations)}\n\n")
            
            if self.violations:
                f.write("="*75 + "\n")
                f.write("VIOLATION DETAILS\n")
                f.write("="*75 + "\n\n")
                
                violation_types = {}
                for v in self.violations:
                    violation_types[v['type']] = violation_types.get(v['type'], 0) + 1
                
                f.write("Summary by Type:\n")
                for vtype, count in violation_types.items():
                    f.write(f"  - {vtype}: {count}\n")
                
                f.write("\n" + "-"*75 + "\n")
                f.write("Chronological Log:\n")
                f.write("-"*75 + "\n\n")
                
                for i, v in enumerate(self.violations, 1):
                    f.write(f"{i}. {v['type']}\n")
                    f.write(f"   Time: {v['time'].strftime('%H:%M:%S')}\n")
                    f.write(f"   Severity: {v['severity']}\n")
                    f.write(f"   Evidence: {v['screenshot']}\n\n")
            else:
                f.write("No violations detected. Excellent behavior!\n")
            
            f.write("="*75 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*75 + "\n")
        
        return report_file
    
    def run(self):
        """Main execution loop"""
        webcam = cv2.VideoCapture(0)
        
        if not webcam.isOpened():
            print("ERROR: Cannot access webcam!")
            return
        
        print("\n" + "="*75)
        print("FINAL PROFESSIONAL AI PROCTORING SYSTEM")
        print("="*75)
        print("✓ System initialized successfully")
        print("✓ Advanced eye filtering enabled (filters nose, mouth, ears)")
        print("✓ Accurate pupil tracking enabled")
        print("\nCalibration Instructions:")
        print("  1. Sit 2-3 feet from camera")
        print("  2. Ensure good lighting on your face")
        print("  3. Look directly at the screen for 2 seconds")
        print("  4. System will calibrate to your baseline gaze")
        print("\nPress ESC to end session and generate report")
        print("="*75 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("ERROR: Cannot read frame from webcam")
                break
            
            # Process frame
            frame, status, color, gaze, detail, violation, eyes = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, status, color, gaze, detail, violation, eyes)
            
            # Display
            cv2.imshow("Final Professional AI Proctoring System", frame)
            
            # ESC to exit
            if cv2.waitKey(1) == 27:
                break
        
        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()
        
        # Generate report
        print("\n" + "="*75)
        print("SESSION ENDED")
        print("="*75)
        report_file = self.generate_report()
        print(f"✓ Report saved: {report_file}")
        print(f"✓ Total violations: {len(self.violations)}")
        if self.violations:
            print(f"✓ Evidence screenshots saved in: violations/")
        print("="*75 + "\n")

if __name__ == "__main__":
    system = FinalProfessionalProctoring()
    system.run()
