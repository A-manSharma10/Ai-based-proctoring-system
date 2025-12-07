"""
ENHANCED PROFESSIONAL AI PROCTORING SYSTEM
With pupil tracking, improved YOLO, and corner violation popups
"""

import cv2
import numpy as np
import os
import datetime
import time
from collections import deque
import urllib.request

class EnhancedProfessionalProctoring:
    def __init__(self):
        print("\n" + "="*80)
        print("PROFESSIONAL EXAM PROCTORING SYSTEM")
        print("="*80)
        print("\nInitializing monitoring system...")
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # YOLO
        print("Loading object detection model...")
        self.yolo_loaded = self.load_yolo()
        
        # Violation tracking
        self.violations = []
        self.violation_counts = {
            'NO_FACE': 0, 'MULTIPLE_FACES': 0, 'LOOKING_AWAY': 0,
            'EYES_CLOSED': 0, 'TOO_CLOSE': 0, 'TOO_FAR': 0,
            'HEAD_TURNED': 0, 'PROFILE_DETECTED': 0,
            'PHONE_DETECTED': 0, 'BOOK_DETECTED': 0,
            'LAPTOP_DETECTED': 0, 'BOTTLE_DETECTED': 0,
            'CUP_DETECTED': 0, 'KEYBOARD_DETECTED': 0,
            'MOUSE_DETECTED': 0
        }
        
        # Thresholds
        self.NO_FACE_THRESHOLD = 45
        self.MULTIPLE_FACE_THRESHOLD = 20
        self.LOOKING_AWAY_THRESHOLD = 60
        self.EYES_CLOSED_THRESHOLD = 90
        self.OBJECT_DETECTION_INTERVAL = 20
        
        # Counters
        self.no_face_counter = 0
        self.multiple_face_counter = 0
        self.looking_away_counter = 0
        self.eyes_closed_counter = 0
        self.frame_count = 0
        
        # Buffers
        self.face_size_buffer = deque(maxlen=10)
        self.eye_detection_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=5)
        self.fps_buffer = deque(maxlen=30)
        
        # Pupil tracking
        self.left_pupil = None
        self.right_pupil = None
        self.gaze_direction = "CENTER"
        
        # Violation popup system
        self.active_popups = []  # List of (text, severity, timestamp)
        self.popup_duration = 3.0  # 3 seconds
        
        # Critical violation counter
        self.critical_violation_count = 0
        self.max_critical_violations = 10
        self.last_critical_time = 0
        self.critical_cooldown = 5.0  # 5 seconds cooldown between critical violations
        
        # Directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        
        # Session
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Calibration
        self.calibrated = False
        self.baseline_face_size = None
        
        # Performance
        self.last_frame_time = time.time()
        self.last_alert_time = 0
        
        # Colors
        self.COLOR_PRIMARY = (0, 255, 150)
        self.COLOR_WARNING = (0, 165, 255)
        self.COLOR_DANGER = (0, 0, 255)
        self.COLOR_CRITICAL = (128, 0, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PANEL = (40, 40, 50)
        
        print("System initialized successfully")
        if self.yolo_loaded:
            print("Object detection: Active")
        print("Eye tracking: Enabled")
        print("\n" + "="*80 + "\n")
    
    def load_yolo(self):
        """Load YOLO"""
        try:
            weights, config, names = 'yolov4-tiny.weights', 'yolov4-tiny.cfg', 'coco.names'
            
            if not os.path.exists(weights):
                print("   ⏳ Downloading weights (~23MB)...")
                urllib.request.urlretrieve(
                    'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
                    weights
                )
            
            if not os.path.exists(config):
                print("   ⏳ Downloading config...")
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                    config
                )
            
            if not os.path.exists(names):
                print("   ⏳ Downloading names...")
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
                    names
                )
            
            self.net = cv2.dnn.readNet(weights, config)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            with open(names, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Enhanced object list (removed remote)
            self.suspicious_objects = {
                'cell phone': 'PHONE',
                'book': 'BOOK',
                'laptop': 'LAPTOP',
                'bottle': 'BOTTLE',
                'cup': 'CUP',
                'keyboard': 'KEYBOARD',
                'mouse': 'MOUSE'
            }
            
            return True
        except Exception as e:
            print(f"   ⚠️  YOLO unavailable: {e}")
            return False
    
    def detect_objects_yolo(self, frame):
        """Enhanced YOLO detection"""
        if not self.yolo_loaded:
            return []
        
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            detected = []
            boxes, confidences, class_ids = [], [], []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Lower threshold for books (harder to detect)
                    threshold = 0.25 if self.classes[class_id] == 'book' else 0.5
                    
                    if confidence > threshold:
                        class_name = self.classes[class_id]
                        if class_name in self.suspicious_objects:
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            width = int(detection[2] * w)
                            height = int(detection[3] * h)
                            x = int(center_x - width / 2)
                            y = int(center_y - height / 2)
                            
                            boxes.append([x, y, width, height])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            if len(boxes) > 0:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, width, height = boxes[i]
                        class_name = self.classes[class_ids[i]]
                        conf = confidences[i]
                        
                        # Draw detection
                        cv2.rectangle(frame, (x, y), (x+width, y+height), self.COLOR_DANGER, 3)
                        label = f"{class_name}: {conf:.0%}"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_DANGER, 2)
                        
                        detected.append(self.suspicious_objects[class_name])
            
            return detected
        except:
            return []
    
    def detect_pupil(self, eye_region):
        """Perfect pupil detection using multiple methods"""
        try:
            if eye_region.size == 0 or eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
                return None
            
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Find darkest point (pupil is darkest)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_eye)
            
            # Method 2: Threshold-based detection
            gray_eye_blur = cv2.GaussianBlur(gray_eye, (5, 5), 0)
            _, threshold = cv2.threshold(gray_eye_blur, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Find the most circular contour near the darkest point
                best_pupil = None
                best_score = 0
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 15 or area > 800:  # Size filter
                        continue
                    
                    # Get bounding box
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    
                    # Check aspect ratio (should be circular)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    if not (0.6 < aspect_ratio < 1.4):
                        continue
                    
                    # Check size
                    if not (8 < w < 50 and 8 < h < 50):
                        continue
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Distance from darkest point
                    center_x, center_y = x + w//2, y + h//2
                    distance = np.sqrt((center_x - min_loc[0])**2 + (center_y - min_loc[1])**2)
                    
                    # Score based on circularity and proximity to darkest point
                    score = circularity * (1.0 / (1.0 + distance/10.0))
                    
                    if score > best_score:
                        best_score = score
                        best_pupil = (center_x, center_y)
                
                if best_pupil and best_score > 0.3:
                    return best_pupil
            
            # Fallback: Use darkest point if it's in a reasonable location
            h, w = gray_eye.shape
            if 0.2 * w < min_loc[0] < 0.8 * w and 0.2 * h < min_loc[1] < 0.8 * h:
                return min_loc
            
            return None
        except:
            return None
    
    def detect_eyes_and_pupils(self, frame, face):
        """Enhanced eye and pupil detection with better visibility"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = cv2.equalizeHist(roi_gray)
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20), maxSize=(80, 80))
        valid_eyes = [e for e in eyes if e[1] < h * 0.6]
        
        # Reset pupil positions
        self.left_pupil = None
        self.right_pupil = None
        
        # Draw eyes and detect pupils
        for i, (ex, ey, ew, eh) in enumerate(valid_eyes[:2]):
            # Draw eye rectangle with glow effect
            cv2.rectangle(roi_color, (ex-1, ey-1), (ex+ew+1, ey+eh+1), (0, 0, 0), 3)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), self.COLOR_PRIMARY, 2)
            
            # Extract eye region for pupil detection
            if ey+eh <= roi_color.shape[0] and ex+ew <= roi_color.shape[1]:
                eye_region = roi_color[ey:ey+eh, ex:ex+ew].copy()
                pupil = self.detect_pupil(eye_region)
                
                if pupil:
                    pupil_x, pupil_y = pupil
                    abs_pupil_x = ex + pupil_x
                    abs_pupil_y = ey + pupil_y
                    
                    # Draw pupil with glow effect (always visible)
                    cv2.circle(roi_color, (abs_pupil_x, abs_pupil_y), 5, (0, 0, 0), -1)  # Black outline
                    cv2.circle(roi_color, (abs_pupil_x, abs_pupil_y), 4, (0, 255, 255), -1)  # Yellow center
                    cv2.circle(roi_color, (abs_pupil_x, abs_pupil_y), 3, (255, 255, 0), -1)  # Bright center
                    
                    # Store pupil positions
                    if i == 0:
                        self.left_pupil = (abs_pupil_x, abs_pupil_y)
                    else:
                        self.right_pupil = (abs_pupil_x, abs_pupil_y)
        
        return valid_eyes
    
    def calculate_gaze_direction(self, eyes, face_width, face_height):
        """Enhanced gaze direction with improved accuracy"""
        if len(eyes) >= 2 and self.left_pupil and self.right_pupil:
            # Calculate average pupil position
            avg_pupil_x = (self.left_pupil[0] + self.right_pupil[0]) / 2
            avg_pupil_y = (self.left_pupil[1] + self.right_pupil[1]) / 2
            
            face_center_x = face_width / 2
            face_center_y = face_height / 2
            
            # Horizontal deviation (more strict for better accuracy)
            h_deviation = (avg_pupil_x - face_center_x) / face_width
            
            # Vertical deviation
            v_deviation = (avg_pupil_y - face_center_y) / face_height
            
            # Store in buffer for smoothing
            self.gaze_buffer.append((h_deviation, v_deviation))
            
            # Calculate average deviation from buffer
            if len(self.gaze_buffer) >= 3:
                avg_h = sum(d[0] for d in self.gaze_buffer) / len(self.gaze_buffer)
                avg_v = sum(d[1] for d in self.gaze_buffer) / len(self.gaze_buffer)
            else:
                avg_h, avg_v = h_deviation, v_deviation
            
            # Determine gaze direction with stricter thresholds
            if abs(avg_h) > abs(avg_v):
                # Horizontal gaze (stricter threshold)
                if avg_h < -0.15:  # More strict (was 0.12)
                    self.gaze_direction = "LEFT"
                    return True, False
                elif avg_h > 0.15:  # More strict (was 0.12)
                    self.gaze_direction = "RIGHT"
                    return True, False
            else:
                # Vertical gaze
                if avg_v < -0.12:  # More strict (was 0.10)
                    self.gaze_direction = "UP"
                    return True, False
                elif avg_v > 0.12:  # More strict (was 0.10)
                    self.gaze_direction = "DOWN"
                    return True, False
            
            # Center
            self.gaze_direction = "CENTER"
            return False, False
        
        if len(eyes) == 0:
            self.gaze_direction = "CLOSED"
            return True, True  # Eyes closed
        if len(eyes) == 1:
            self.gaze_direction = "AWAY"
            return True, False  # Looking away
        
        return False, False
    
    def add_popup(self, text, severity):
        """Add violation popup with suggestions"""
        # Create suggestion based on violation type
        suggestions = {
            'NO FACE': 'Return to camera view',
            'MULTIPLE FACES': 'Only one person allowed',
            'LOOKING AWAY': 'Look straight at screen',
            'EYES CLOSED': 'Keep eyes open',
            'TOO CLOSE': 'Move back from camera',
            'TOO FAR': 'Move closer to camera',
            'HEAD TURNED': 'Face the camera',
            'PROFILE DETECTED': 'Face forward',
            'PHONE DETECTED': 'Remove phone immediately',
            'BOOK DETECTED': 'Remove book immediately',
            'LAPTOP DETECTED': 'Remove laptop immediately',
            'BOTTLE DETECTED': 'Remove bottle',
            'CUP DETECTED': 'Remove cup',
            'KEYBOARD DETECTED': 'Remove keyboard',
            'MOUSE DETECTED': 'Remove mouse'
        }
        
        suggestion = suggestions.get(text, 'Follow exam rules')
        
        self.active_popups.append({
            'text': text,
            'suggestion': suggestion,
            'severity': severity,
            'timestamp': time.time()
        })
    
    def draw_popups(self, frame):
        """Draw bottom-right corner violation popups"""
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Remove expired popups
        self.active_popups = [p for p in self.active_popups 
                             if current_time - p['timestamp'] < self.popup_duration]
        
        # Draw active popups in bottom-right corner
        y_offset = h - 80  # Start from bottom
        for popup in self.active_popups:
            # Color based on severity
            if popup['severity'] == 'CRITICAL':
                color = self.COLOR_CRITICAL
            elif popup['severity'] == 'HIGH':
                color = self.COLOR_DANGER
            else:
                color = self.COLOR_WARNING
            
            # Calculate popup dimensions
            text = popup['text']
            suggestion = popup['suggestion']
            
            # Calculate size for both lines
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (sug_w, sug_h), _ = cv2.getTextSize(suggestion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            popup_w = max(text_w, sug_w) + 50
            popup_h = text_h + sug_h + 40
            popup_x = w - popup_w - 20
            popup_y = y_offset - popup_h
            
            # Draw popup background with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), color, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (popup_x, popup_y), (popup_x + popup_w, popup_y + popup_h), self.COLOR_TEXT, 2)
            
            # Draw warning icon and text (using ASCII only)
            cv2.putText(frame, "!", (popup_x + 15, popup_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
            cv2.putText(frame, text, (popup_x + 40, popup_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)
            
            # Draw suggestion
            cv2.putText(frame, suggestion, (popup_x + 15, popup_y + popup_h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
            
            y_offset = popup_y - 10  # Move up for next popup
        
        return frame

    
    def calculate_fps(self):
        """Calculate FPS"""
        current = time.time()
        fps = 1.0 / (current - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = current
        self.fps_buffer.append(fps)
        return sum(self.fps_buffer) / len(self.fps_buffer)
    
    def check_distance(self, face_size):
        """Check distance"""
        if not self.baseline_face_size:
            return None
        self.face_size_buffer.append(face_size)
        ratio = (sum(self.face_size_buffer)/len(self.face_size_buffer)) / self.baseline_face_size
        if ratio > 1.8:
            return 'TOO_CLOSE'
        elif ratio < 0.3:
            return 'TOO_FAR'
        return None
    
    def play_alert(self):
        """Alert sound"""
        if time.time() - self.last_alert_time > 5:
            try:
                import winsound
                winsound.Beep(1200, 250)
            except:
                pass
            self.last_alert_time = time.time()
    
    def log_violation(self, vtype, severity, details=""):
        """Log violation with cooldown for critical violations"""
        current_time = time.time()
        
        # Check cooldown for critical violations
        if severity == 'CRITICAL':
            if current_time - self.last_critical_time < self.critical_cooldown:
                # Still in cooldown period, don't count this critical violation
                print(f"   Cooldown active ({self.critical_cooldown - (current_time - self.last_critical_time):.1f}s remaining)")
                return
            else:
                # Cooldown expired, count this critical violation
                self.last_critical_time = current_time
                self.critical_violation_count += 1
                print(f"VIOLATION #{len(self.violations) + 1}: {vtype} [{severity}]")
                print(f"   CRITICAL COUNT: {self.critical_violation_count}/{self.max_critical_violations}")
                print(f"   You have {self.critical_cooldown} seconds to resolve this issue")
        else:
            print(f"VIOLATION #{len(self.violations) + 1}: {vtype} [{severity}]")
        
        self.violations.append({
            'type': vtype,
            'severity': severity,
            'timestamp': datetime.datetime.now(),
            'details': details
        })
        self.violation_counts[vtype] += 1
        
        # Add popup
        self.add_popup(vtype.replace('_', ' '), severity)
        
        if severity in ['HIGH', 'CRITICAL']:
            self.play_alert()
    
    def save_screenshot(self, frame, vtype):
        """Save screenshot"""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f"violations/{self.session_id}_{vtype}_{ts}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    def calibrate(self, face):
        """Calibrate"""
        if not self.calibrated:
            x, y, w, h = face
            self.baseline_face_size = w * h
            self.calibrated = True
            print("Calibration complete - Monitoring started\n")
    
    def draw_ui(self, frame, status, objects, fps):
        """Draw professional UI"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 100), self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "EXAM PROCTORING", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, self.COLOR_PRIMARY, 3)
        cv2.putText(frame, "MONITORING SYSTEM", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        
        # Status indicator
        status_color = self.COLOR_PRIMARY if self.calibrated else self.COLOR_WARNING
        cv2.circle(frame, (w-50, 50), 15, status_color, -1)
        cv2.putText(frame, "LIVE" if self.calibrated else "SETUP", (w-140, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Left panel - Statistics
        panel_w, panel_h = 320, h - 120
        panel_y = 110
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (panel_w, panel_y + panel_h), self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, panel_y), (panel_w, panel_y + panel_h), self.COLOR_PRIMARY, 2)
        
        cv2.putText(frame, "SESSION STATS", (25, panel_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.COLOR_PRIMARY, 2)
        cv2.line(frame, (25, panel_y + 45), (panel_w - 15, panel_y + 45), self.COLOR_PRIMARY, 2)
        
        y_pos = panel_y + 75
        
        # Time
        elapsed = str(datetime.datetime.now() - self.session_start).split('.')[0]
        cv2.putText(frame, f"Duration: {elapsed}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        y_pos += 35
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        y_pos += 35
        
        # Gaze direction
        gaze_color = self.COLOR_PRIMARY if self.gaze_direction == "CENTER" else self.COLOR_WARNING
        cv2.putText(frame, f"Gaze: {self.gaze_direction}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        y_pos += 35
        
        # Violations
        viol_color = self.COLOR_PRIMARY if len(self.violations) == 0 else self.COLOR_DANGER
        cv2.putText(frame, f"Violations: {len(self.violations)}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, viol_color, 1)
        y_pos += 35
        
        # Critical violations counter
        crit_color = self.COLOR_DANGER if self.critical_violation_count > 5 else self.COLOR_WARNING
        cv2.putText(frame, f"Critical: {self.critical_violation_count}/{self.max_critical_violations}", 
                   (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, crit_color, 1)
        y_pos += 35
        
        # Cooldown indicator
        current_time = time.time()
        if current_time - self.last_critical_time < self.critical_cooldown:
            remaining = self.critical_cooldown - (current_time - self.last_critical_time)
            cv2.putText(frame, f"Cooldown: {remaining:.1f}s", 
                       (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 1)
            y_pos += 35
        
        y_pos += 10
        
        # Violation breakdown
        if len(self.violations) > 0:
            cv2.putText(frame, "BREAKDOWN:", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 1)
            y_pos += 30
            for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0 and y_pos < panel_y + panel_h - 30:
                    # Use simple bullet point (no emoji)
                    cv2.putText(frame, f"- {vtype}: {count}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
                    y_pos += 25
        
        # Right panel - Detection
        right_x = w - panel_w - 10
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (right_x, panel_y), (w - 10, panel_y + 250), self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (right_x, panel_y), (w - 10, panel_y + 250), self.COLOR_PRIMARY, 2)
        
        cv2.putText(frame, "DETECTION", (right_x + 15, panel_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.COLOR_PRIMARY, 2)
        cv2.line(frame, (right_x + 15, panel_y + 45), (w - 25, panel_y + 45), self.COLOR_PRIMARY, 2)
        
        y_pos = panel_y + 80
        
        # Detection status
        yolo_text = "ACTIVE" if self.yolo_loaded else "INACTIVE"
        yolo_color = self.COLOR_PRIMARY if self.yolo_loaded else self.COLOR_WARNING
        cv2.putText(frame, f"YOLO: {yolo_text}", (right_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 1)
        y_pos += 35
        
        cv2.putText(frame, "Face: ON", (right_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_PRIMARY, 1)
        y_pos += 35
        
        cv2.putText(frame, "Pupil: ON", (right_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_PRIMARY, 1)
        y_pos += 45
        
        # Objects
        if objects:
            cv2.putText(frame, "OBJECTS:", (right_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_DANGER, 1)
            y_pos += 30
            for obj in set(objects):
                # Use simple ! instead of emoji
                cv2.putText(frame, f"! {obj}", (right_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_DANGER, 1)
                y_pos += 25
        else:
            cv2.putText(frame, "Clear", (right_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_PRIMARY, 1)
        
        # Bottom bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, status, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
        cv2.putText(frame, "Press ESC to End", (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 1)
        
        return frame
    
    def generate_report(self):
        """Generate report"""
        path = f"reports/session_{self.session_id}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED PROFESSIONAL PROCTORING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            end = datetime.datetime.now()
            f.write(f"End: {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {str(end - self.session_start).split('.')[0]}\n")
            f.write(f"YOLO: {'Active' if self.yolo_loaded else 'Inactive'}\n")
            f.write(f"Pupil Tracking: Enabled\n\n")
            f.write(f"Total Violations: {len(self.violations)}\n\n")
            for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    f.write(f"  {vtype}: {count}\n")
            f.write("\n" + "="*80 + "\n")
            for i, v in enumerate(self.violations, 1):
                f.write(f"{i}. [{v['timestamp'].strftime('%H:%M:%S')}] {v['type']} - {v['severity']}\n")
            f.write("="*80 + "\n")
        return path
    
    def run(self):
        """Main loop"""
        print("Starting proctoring session...")
        print("Calibrating... Please look at screen for 3 seconds\n")
        
        calibration_frames, calibration_needed = 0, 90
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            fps = self.calculate_fps()
            
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
            profiles = self.profile_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
            
            status = "Initializing..."
            detected_objects = []
            
            # Calibration
            if not self.calibrated:
                if len(faces) == 1:
                    calibration_frames += 1
                    status = f"Calibrating... {calibration_frames}/{calibration_needed}"
                    progress = int((calibration_frames / calibration_needed) * w)
                    cv2.rectangle(frame, (0, h-30), (progress, h), self.COLOR_PRIMARY, -1)
                    if calibration_frames >= calibration_needed:
                        self.calibrate(faces[0])
                else:
                    status = "Waiting for single face..."
                    calibration_frames = 0
            
            # Monitoring
            if self.calibrated:
                status = "Monitoring Active"
                
                # YOLO detection
                if self.frame_count % self.OBJECT_DETECTION_INTERVAL == 0:
                    detected_objects = self.detect_objects_yolo(frame)
                    for obj in detected_objects:
                        self.log_violation(f"{obj}_DETECTED", 'CRITICAL', f"{obj} detected")
                        self.save_screenshot(frame, f"{obj}_DETECTED")
                
                # Face analysis
                if len(faces) == 0 and len(profiles) == 0:
                    self.no_face_counter += 1
                    if self.no_face_counter >= self.NO_FACE_THRESHOLD:
                        self.log_violation('NO_FACE', 'CRITICAL', f"No face for {self.no_face_counter} frames")
                        self.save_screenshot(frame, 'NO_FACE')
                        self.no_face_counter = 0
                else:
                    self.no_face_counter = 0
                
                if len(profiles) > 0 and len(faces) == 0:
                    self.log_violation('PROFILE_DETECTED', 'MEDIUM', "Profile detected")
                    self.save_screenshot(frame, 'PROFILE_DETECTED')
                    for (x, y, w, h) in profiles:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), self.COLOR_WARNING, 3)
                
                if len(faces) > 1:
                    self.multiple_face_counter += 1
                    if self.multiple_face_counter >= self.MULTIPLE_FACE_THRESHOLD:
                        self.log_violation('MULTIPLE_FACES', 'CRITICAL', f"{len(faces)} faces")
                        self.save_screenshot(frame, 'MULTIPLE_FACES')
                        self.multiple_face_counter = 0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), self.COLOR_DANGER, 3)
                        cv2.putText(frame, "UNAUTHORIZED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_DANGER, 2)
                else:
                    self.multiple_face_counter = 0
                
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), self.COLOR_PRIMARY, 2)
                    cv2.putText(frame, "AUTHORIZED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_PRIMARY, 2)
                    
                    # Distance
                    dist_status = self.check_distance(w * h)
                    if dist_status:
                        self.log_violation(dist_status, 'MEDIUM', f"Distance: {dist_status}")
                        self.save_screenshot(frame, dist_status)
                    
                    # Eyes and pupils
                    eyes = self.detect_eyes_and_pupils(frame, faces[0])
                    looking_away, eyes_closed = self.calculate_gaze_direction(eyes, w, h)
                    
                    if eyes_closed:
                        self.eyes_closed_counter += 1
                        if self.eyes_closed_counter >= self.EYES_CLOSED_THRESHOLD:
                            self.log_violation('EYES_CLOSED', 'HIGH', f"{self.eyes_closed_counter} frames")
                            self.save_screenshot(frame, 'EYES_CLOSED')
                            self.eyes_closed_counter = 0
                    else:
                        self.eyes_closed_counter = 0
                    
                    if looking_away and not eyes_closed:
                        self.looking_away_counter += 1
                        if self.looking_away_counter >= self.LOOKING_AWAY_THRESHOLD:
                            self.log_violation('LOOKING_AWAY', 'MEDIUM', f"Gaze: {self.gaze_direction}")
                            self.save_screenshot(frame, 'LOOKING_AWAY')
                            self.looking_away_counter = 0
                    else:
                        self.looking_away_counter = 0
            
            # Draw UI
            frame = self.draw_ui(frame, status, detected_objects, fps)
            
            # Draw popups (corner notifications)
            frame = self.draw_popups(frame)
            
            cv2.imshow('Enhanced Professional Proctoring System', frame)
            
            # Check if max critical violations reached
            if self.critical_violation_count >= self.max_critical_violations:
                print("\n" + "="*80)
                print("MAXIMUM CRITICAL VIOLATIONS REACHED!")
                print("EXAM TERMINATED AUTOMATICALLY")
                print("="*80)
                
                # Show final termination popup
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 128), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                
                cv2.putText(frame, "EXAM TERMINATED", (w//2 - 200, h//2 - 60),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, "Maximum violations reached", (w//2 - 180, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Critical violations: {self.critical_violation_count}/{self.max_critical_violations}", 
                           (w//2 - 150, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Contact exam administrator", (w//2 - 160, h//2 + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Enhanced Professional Proctoring System', frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                break
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*80)
        print("SESSION ENDED")
        print("="*80)
        report = self.generate_report()
        print(f"\nReport saved: {report}")
        print(f"Total violations: {len(self.violations)}")
        if len(self.violations) > 0:
            print("\nViolation Summary:")
            for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                if count > 0:
                    print(f"  - {vtype}: {count}")
        print("\n" + "="*80 + "\n")


def main():
    try:
        system = EnhancedProfessionalProctoring()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
