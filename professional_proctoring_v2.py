"""
PROFESSIONAL AI PROCTORING SYSTEM
Features:
- Accurate pupil detection and gaze tracking using MediaPipe
- Distance detection (too close/too far)
- Real-time warning messages on screen
- Visual gaze alert bar
- Audio alerts
- Professional UI with detailed status
- Session reports and violation screenshots
"""
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import os
import winsound
from collections import deque

class ProfessionalProctoring:
    def __init__(self):
        # MediaPipe Face Mesh for accurate tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye and iris landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Create folders
        os.makedirs('violations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Tracking counters
        self.looking_away_frames = 0
        self.no_face_frames = 0
        self.multiple_face_frames = 0
        self.eyes_closed_frames = 0
        self.too_close_frames = 0
        self.too_far_frames = 0
        
        # Thresholds (in frames at ~30fps)
        self.LOOKING_AWAY_LIMIT = 60  # 2 seconds
        self.NO_FACE_LIMIT = 45  # 1.5 seconds
        self.MULTIPLE_FACE_LIMIT = 30  # 1 second
        self.EYES_CLOSED_LIMIT = 90  # 3 seconds
        self.DISTANCE_LIMIT = 45  # 1.5 seconds
        
        # Distance thresholds (based on face size)
        self.OPTIMAL_FACE_SIZE = 0.25  # 25% of frame height
        self.TOO_CLOSE_THRESHOLD = 0.40  # 40% of frame height
        self.TOO_FAR_THRESHOLD = 0.15  # 15% of frame height
        
        # Gaze thresholds
        self.GAZE_THRESHOLD = 0.020  # Sensitivity for gaze detection
        
        # Session tracking
        self.violations = []
        self.session_start = datetime.now()
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts
        
        # Stats
        self.total_frames = 0
        self.normal_frames = 0
        
        # Smoothing for gaze detection
        self.gaze_history = deque(maxlen=10)
        
        # Current warnings
        self.current_warnings = []
        
    def play_alert(self, frequency=1000, duration=200):
        """Play warning sound"""
        try:
            winsound.Beep(frequency, duration)
        except:
            pass
    
    def capture_violation(self, frame, violation_type):
        """Save screenshot of violation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
            
            # Different alert sounds for different severities
            if severity == "HIGH":
                self.play_alert(1500, 300)
            elif severity == "MEDIUM":
                self.play_alert(1000, 200)
            else:
                self.play_alert(800, 150)
            
            self.last_alert_time = current_time
            print(f"⚠️  VIOLATION: {violation_type} [{severity}]")
    
    def calculate_face_size(self, landmarks, frame_shape):
        """Calculate face size relative to frame"""
        h, w = frame_shape[:2]
        
        # Get face oval points
        face_points = []
        for idx in self.FACE_OVAL:
            x = landmarks[idx].x * w
            y = landmarks[idx].y * h
            face_points.append([x, y])
        
        face_points = np.array(face_points, dtype=np.int32)
        
        # Calculate bounding box
        x, y, fw, fh = cv2.boundingRect(face_points)
        
        # Face size as percentage of frame height
        face_size_ratio = fh / h
        
        return face_size_ratio, (x, y, fw, fh)
    
    def detect_gaze_direction(self, landmarks, frame_shape):
        """Detect gaze direction using iris position"""
        h, w = frame_shape[:2]
        
        # Left eye analysis
        left_iris_x = np.mean([landmarks[i].x for i in self.LEFT_IRIS])
        left_iris_y = np.mean([landmarks[i].y for i in self.LEFT_IRIS])
        left_eye_x = np.mean([landmarks[i].x for i in self.LEFT_EYE])
        left_eye_y = np.mean([landmarks[i].y for i in self.LEFT_EYE])
        
        # Right eye analysis
        right_iris_x = np.mean([landmarks[i].x for i in self.RIGHT_IRIS])
        right_iris_y = np.mean([landmarks[i].y for i in self.RIGHT_IRIS])
        right_eye_x = np.mean([landmarks[i].x for i in self.RIGHT_EYE])
        right_eye_y = np.mean([landmarks[i].y for i in self.RIGHT_EYE])
        
        # Calculate horizontal deviation
        left_diff = left_iris_x - left_eye_x
        right_diff = right_iris_x - right_eye_x
        horizontal_gaze = (left_diff + right_diff) / 2
        
        # Calculate vertical deviation
        left_v_diff = left_iris_y - left_eye_y
        right_v_diff = right_iris_y - right_eye_y
        vertical_gaze = (left_v_diff + right_v_diff) / 2
        
        # Smooth gaze detection
        self.gaze_history.append((horizontal_gaze, vertical_gaze))
        avg_h_gaze = np.mean([g[0] for g in self.gaze_history])
        avg_v_gaze = np.mean([g[1] for g in self.gaze_history])
        
        # Determine gaze direction
        gaze_status = "CENTER"
        looking_away = False
        
        if abs(avg_h_gaze) > self.GAZE_THRESHOLD or abs(avg_v_gaze) > self.GAZE_THRESHOLD * 1.5:
            looking_away = True
            if avg_h_gaze < -self.GAZE_THRESHOLD:
                gaze_status = "RIGHT"
            elif avg_h_gaze > self.GAZE_THRESHOLD:
                gaze_status = "LEFT"
            elif avg_v_gaze < -self.GAZE_THRESHOLD * 1.5:
                gaze_status = "UP"
            elif avg_v_gaze > self.GAZE_THRESHOLD * 1.5:
                gaze_status = "DOWN"
        
        # Draw iris positions
        left_iris_px = (int(left_iris_x * w), int(left_iris_y * h))
        right_iris_px = (int(right_iris_x * w), int(right_iris_y * h))
        
        return looking_away, gaze_status, (avg_h_gaze, avg_v_gaze), left_iris_px, right_iris_px
    
    def detect_eyes_closed(self, landmarks):
        """Detect if eyes are closed using eye aspect ratio"""
        # Left eye vertical distances
        left_top = landmarks[386].y
        left_bottom = landmarks[374].y
        left_height = abs(left_top - left_bottom)
        
        # Right eye vertical distances
        right_top = landmarks[159].y
        right_bottom = landmarks[145].y
        right_height = abs(right_top - right_bottom)
        
        # Average eye opening
        avg_height = (left_height + right_height) / 2
        
        # Threshold for closed eyes
        return avg_height < 0.008

    
    def analyze_frame(self, frame):
        """Main analysis function"""
        self.total_frames += 1
        self.current_warnings = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        alert_status = "✓ MONITORING - NORMAL"
        alert_color = (0, 255, 0)
        violation_detected = False
        gaze_percentage = 0
        gaze_status = "UNKNOWN"
        iris_positions = None
        
        # Check for faces
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            
            # Multiple faces check
            if num_faces > 1:
                self.multiple_face_frames += 1
                self.current_warnings.append("⚠️ MULTIPLE PEOPLE DETECTED")
                if self.multiple_face_frames > self.MULTIPLE_FACE_LIMIT:
                    alert_status = "🚨 VIOLATION: MULTIPLE PEOPLE"
                    alert_color = (0, 0, 255)
                    violation_detected = True
                    self.add_violation(frame, "MULTIPLE_FACES", "HIGH")
                    self.multiple_face_frames = 0
            else:
                self.multiple_face_frames = 0
                
                # Analyze single face
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                
                # Check distance
                face_size, face_bbox = self.calculate_face_size(landmarks, frame.shape)
                
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
                
                # Draw face boundary
                x, y, fw, fh = face_bbox
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
                
                # Check if eyes are closed
                eyes_closed = self.detect_eyes_closed(landmarks)
                if eyes_closed:
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
                
                # Gaze tracking
                if not eyes_closed:
                    looking_away, gaze_status, gaze_values, left_iris, right_iris = self.detect_gaze_direction(landmarks, frame.shape)
                    iris_positions = (left_iris, right_iris)
                    
                    # Draw iris
                    cv2.circle(frame, left_iris, 3, (0, 255, 255), -1)
                    cv2.circle(frame, right_iris, 3, (0, 255, 255), -1)
                    
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
            
            # Reset no face counter
            self.no_face_frames = 0
            
        else:
            # No face detected
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
    
    def draw_professional_ui(self, frame, status, color, gaze_percentage, gaze_status):
        """Draw professional UI overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Main status
        cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Session info
        elapsed = datetime.now() - self.session_start
        elapsed_str = str(elapsed).split('.')[0]
        cv2.putText(frame, f"Session: {elapsed_str}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations: {len(self.violations)}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze: {gaze_status}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Compliance score
        if self.total_frames > 0:
            compliance = int((self.normal_frames / self.total_frames) * 100)
            compliance_color = (0, 255, 0) if compliance > 90 else (0, 165, 255) if compliance > 70 else (0, 0, 255)
            cv2.putText(frame, f"Compliance: {compliance}%", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, compliance_color, 2)
        
        # Violation counter (top right)
        if len(self.violations) > 0:
            cv2.rectangle(frame, (w-200, 10), (w-10, 70), (0, 0, 200), -1)
            cv2.putText(frame, f"VIOLATIONS", (w-185, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{len(self.violations)}", (w-110,