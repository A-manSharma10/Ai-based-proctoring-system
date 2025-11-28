"""
ULTIMATE YOLO-POWERED AI PROCTORING SYSTEM
The most accurate and comprehensive proctoring solution with:
- YOLOv4-tiny for precise object detection
- Multi-stage calibration
- Adaptive thresholds
- Maximum accuracy
"""

import cv2
import numpy as np
import os
import datetime
import time
from collections import deque
import urllib.request

class UltimateYOLOProctoringSystem:
    def __init__(self):
        print("\n" + "="*70)
        print("ULTIMATE YOLO-POWERED AI PROCTORING SYSTEM")
        print("="*70)
        print("Initializing system...")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Load YOLO
        self.yolo_loaded = self.load_yolo()
        
        # Violation tracking
        self.violations = []
        self.violation_counts = {
            'NO_FACE': 0,
            'MULTIPLE_FACES': 0,
            'LOOKING_AWAY': 0,
            'EYES_CLOSED': 0,
            'TOO_CLOSE': 0,
            'TOO_FAR': 0,
            'HEAD_TURNED': 0,
            'PROFILE_DETECTED': 0,
            'PHONE_DETECTED': 0,
            'BOOK_DETECTED': 0,
            'LAPTOP_DETECTED': 0,
            'BOTTLE_DETECTED': 0,
            'HAND_NEAR_FACE': 0
        }
        
        # Enhanced thresholds
        self.NO_FACE_THRESHOLD = 45
        self.MULTIPLE_FACE_THRESHOLD = 20
        self.LOOKING_AWAY_THRESHOLD = 60
        self.EYES_CLOSED_THRESHOLD = 90
        self.HEAD_TURNED_THRESHOLD = 45
        self.OBJECT_DETECTION_INTERVAL = 20
        
        # Counters
        self.no_face_counter = 0
        self.multiple_face_counter = 0
        self.looking_away_counter = 0
        self.eyes_closed_counter = 0
        self.head_turned_counter = 0
        self.frame_count = 0
        
        # Smoothing buffers
        self.face_size_buffer = deque(maxlen=10)
        self.eye_detection_buffer = deque(maxlen=5)
        self.gaze_buffer = deque(maxlen=5)
        
        # Directories
        os.makedirs('reports', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        
        # Session info
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Calibration
        self.calibrated = False
        self.baseline_face_size = None
        self.baseline_face_width = None
        self.baseline_eye_distance = None
        
        # Alert system
        self.last_alert_time = 0
        self.alert_cooldown = 5
        
        # Performance
        self.fps_buffer = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        print("✓ System initialized successfully")
        if self.yolo_loaded:
            print("✓ YOLO object detection: ACTIVE")
        else:
            print("⚠️  YOLO unavailable - using fallback detection")
        print("\nPress ESC to end session")
        print("="*70 + "\n")
    
    def load_yolo(self):
        """Load YOLO model with automatic download"""
        try:
            weights_file = 'yolov4-tiny.weights'
            config_file = 'yolov4-tiny.cfg'
            names_file = 'coco.names'
            
            # Download files if not present
            if not os.path.exists(weights_file):
                print("⏳ Downloading YOLOv4-tiny weights (~23MB)...")
                try:
                    urllib.request.urlretrieve(
                        'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
                        weights_file
                    )
                    print("✓ Weights downloaded")
                except Exception as e:
                    print(f"⚠️  Download failed: {e}")
                    return False
            
            if not os.path.exists(config_file):
                print("⏳ Downloading YOLOv4-tiny config...")
                try:
                    urllib.request.urlretrieve(
                        'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                        config_file
                    )
                    print("✓ Config downloaded")
                except Exception as e:
                    print(f"⚠️  Download failed: {e}")
                    return False
            
            if not os.path.exists(names_file):
                print("⏳ Downloading COCO names...")
                try:
                    urllib.request.urlretrieve(
                        'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
                        names_file
                    )
                    print("✓ Names downloaded")
                except Exception as e:
                    print(f"⚠️  Download failed: {e}")
                    return False
            
            # Load YOLO
            self.net = cv2.dnn.readNet(weights_file, config_file)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load class names
            with open(names_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Get output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Suspicious objects to detect
            self.suspicious_objects = {
                'cell phone': 'PHONE',
                'book': 'BOOK',
                'laptop': 'LAPTOP',
                'bottle': 'BOTTLE',
                'cup': 'CUP',
                'keyboard': 'KEYBOARD',
                'mouse': 'MOUSE'
            }
            
            print("✓ YOLO model loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  YOLO loading failed: {e}")
            return False
    
    def detect_objects_yolo(self, frame):
        """Detect objects using YOLO"""
        if not self.yolo_loaded:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Prepare image for YOLO
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            detected_objects = []
            boxes = []
            confidences = []
            class_ids = []
            
            # Process detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:  # 50% confidence threshold
                        class_name = self.classes[class_id]
                        
                        if class_name in self.suspicious_objects:
                            # Get bounding box
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            # Apply non-maximum suppression
            if len(boxes) > 0:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        class_name = self.classes[class_ids[i]]
                        confidence = confidences[i]
                        
                        # Draw detection
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        detected_objects.append(self.suspicious_objects[class_name])
            
            return detected_objects
            
        except Exception as e:
            return []
    
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = current_time
        self.fps_buffer.append(fps)
        return sum(self.fps_buffer) / len(self.fps_buffer)
    
    def smooth_face_size(self, face_size):
        """Smooth face size"""
        self.face_size_buffer.append(face_size)
        return sum(self.face_size_buffer) / len(self.face_size_buffer)
    
    def detect_eyes_enhanced(self, frame, face):
        """Enhanced eye detection"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = cv2.equalizeHist(roi_gray)
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20), maxSize=(80, 80))
        
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            if ey < h * 0.6:
                valid_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return valid_eyes
    
    def detect_gaze_advanced(self, frame, face):
        """Advanced gaze detection"""
        eyes = self.detect_eyes_enhanced(frame, face)
        x, y, w, h = face
        
        if len(eyes) == 0:
            self.eye_detection_buffer.append('none')
            if self.eye_detection_buffer.count('none') >= 3:
                return True, True
            return False, False
        
        if len(eyes) == 1:
            self.eye_detection_buffer.append('one')
            if self.eye_detection_buffer.count('one') >= 3:
                return True, False
            return False, False
        
        if len(eyes) >= 2:
            self.eye_detection_buffer.append('two')
            eye_centers = [(ex + ew//2) for (ex, ey, ew, eh) in eyes[:2]]
            face_center = w // 2
            eyes_center = sum(eye_centers) / 2
            deviation_ratio = abs(eyes_center - face_center) / w
            
            self.gaze_buffer.append(deviation_ratio)
            avg_deviation = sum(self.gaze_buffer) / len(self.gaze_buffer)
            
            if avg_deviation > 0.3:
                return True, False
        
        return False, False
    
    def check_distance(self, face_size):
        """Check distance"""
        if self.baseline_face_size is None:
            return None, 1.0
        
        smoothed_size = self.smooth_face_size(face_size)
        ratio = smoothed_size / self.baseline_face_size
        
        if ratio > 1.8:
            return 'TOO_CLOSE', ratio
        elif ratio < 0.3:
            return 'TOO_FAR', ratio
        
        return None, ratio
