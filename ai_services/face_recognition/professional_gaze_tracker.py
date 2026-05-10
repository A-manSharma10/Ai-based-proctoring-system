"""
Gaze and Pupil Tracking Module
Implements attention monitoring using iris landmarks and head pose estimation.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import time
import logging

logger = logging.getLogger(__name__)


class GazeTracker:
    """
    Gaze tracker utilizing iris landmarks and head orientation.
    """
    
    def __init__(self,
                 looking_away_threshold: float = 2.0,
                 window_size: int = 50,
                 blink_detection: bool = True):
        """
        Initialize gaze tracker parameters.
        
        Args:
            looking_away_threshold: Seconds before flagging
            window_size: Frames for temporal smoothing
            blink_detection: Enable blink rate monitoring
        """
        self.looking_away_threshold = looking_away_threshold
        self.window_size = window_size
        self.blink_detection = blink_detection
        
        # Initialize Face Mesh for iris tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        
        # Tracking state
        self.gaze_history = deque(maxlen=window_size)
        self.blink_history = deque(maxlen=100)  # 10 seconds
        self.looking_away_start = None
        self.last_violation = 0
        self.violation_cooldown = 10.0
        
        # Iris landmark indices (MediaPipe 468-point model)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Eye landmark indices
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Face oval for head pose
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
    def calculate_eye_aspect_ratio(self, eye_landmarks, landmarks, image_shape) -> float:
        """
        Calculate Eye Aspect Ratio for blink detection.
        
        Args:
            eye_landmarks: Indices of eye landmarks
            landmarks: All face landmarks
            image_shape: Image dimensions
            
        Returns:
            Eye aspect ratio
        """
        try:
            h, w = image_shape[:2]
            
            # Get eye points
            points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                              for i in eye_landmarks])
            
            # Calculate vertical distances
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance
            h_dist = np.linalg.norm(points[0] - points[3])
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h_dist)
            
            return ear
            
        except Exception as e:
            return 0.3  # Default value
    
    def detect_blink(self, landmarks, image_shape) -> bool:
        """
        Detect if eyes are blinking.
        
        Args:
            landmarks: Face landmarks
            image_shape: Image dimensions
            
        Returns:
            True if blinking
        """
        try:
            left_ear = self.calculate_eye_aspect_ratio(self.LEFT_EYE, landmarks, image_shape)
            right_ear = self.calculate_eye_aspect_ratio(self.RIGHT_EYE, landmarks, image_shape)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Blink threshold
            return avg_ear < 0.2
            
        except Exception as e:
            return False
    
    def calculate_iris_position(self, iris_landmarks, eye_landmarks, landmarks, image_shape) -> Tuple[float, float]:
        """
        Calculate iris position relative to eye.
        
        Args:
            iris_landmarks: Iris landmark indices
            eye_landmarks: Eye landmark indices
            landmarks: All face landmarks
            image_shape: Image dimensions
            
        Returns:
            (horizontal_ratio, vertical_ratio) where 0.5 is center
        """
        try:
            h, w = image_shape[:2]
            
            # Get iris center
            iris_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                   for i in iris_landmarks])
            iris_center = np.mean(iris_points, axis=0)
            
            # Get eye corners
            eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                                  for i in eye_landmarks])
            
            # Calculate eye bounding box
            eye_left = np.min(eye_points[:, 0])
            eye_right = np.max(eye_points[:, 0])
            eye_top = np.min(eye_points[:, 1])
            eye_bottom = np.max(eye_points[:, 1])
            
            # Calculate ratios
            h_ratio = (iris_center[0] - eye_left) / (eye_right - eye_left) if eye_right > eye_left else 0.5
            v_ratio = (iris_center[1] - eye_top) / (eye_bottom - eye_top) if eye_bottom > eye_top else 0.5
            
            return h_ratio, v_ratio
            
        except Exception as e:
            return 0.5, 0.5
    
    def calculate_head_pose(self, landmarks, image_shape) -> Dict[str, float]:
        """
        Calculate head pose angles (yaw, pitch, roll).
        
        Args:
            landmarks: Face landmarks
            image_shape: Image dimensions
            
        Returns:
            Dictionary with yaw, pitch, roll
        """
        try:
            h, w = image_shape[:2]
            
            # 3D model points
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye corner
                (225.0, 170.0, -135.0),      # Right eye corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            # 2D image points
            image_points = np.array([
                (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
                (landmarks[152].x * w, landmarks[152].y * h),  # Chin
                (landmarks[33].x * w, landmarks[33].y * h),    # Left eye corner
                (landmarks[263].x * w, landmarks[263].y * h),  # Right eye corner
                (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth corner
                (landmarks[291].x * w, landmarks[291].y * h)   # Right mouth corner
            ], dtype="double")
            
            # Camera matrix
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            # Distortion coefficients
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            
            if sy > 1e-6:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0
            
            # Convert to degrees
            return {
                'yaw': float(np.degrees(yaw)),
                'pitch': float(np.degrees(pitch)),
                'roll': float(np.degrees(roll))
            }
            
        except Exception as e:
            logger.warning(f"Head pose calculation failed: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def calculate_gaze_direction(self, landmarks, image_shape) -> Dict[str, Any]:
        """
        Calculate gaze direction using iris position and head pose.
        
        Args:
            landmarks: Face landmarks
            image_shape: Image dimensions
            
        Returns:
            Gaze analysis results
        """
        # Calculate iris positions
        left_h, left_v = self.calculate_iris_position(
            self.LEFT_IRIS, self.LEFT_EYE, landmarks, image_shape
        )
        right_h, right_v = self.calculate_iris_position(
            self.RIGHT_IRIS, self.RIGHT_EYE, landmarks, image_shape
        )
        
        # Average iris position
        avg_h = (left_h + right_h) / 2.0
        avg_v = (left_v + right_v) / 2.0
        
        # Calculate head pose
        head_pose = self.calculate_head_pose(landmarks, image_shape)
        
        # Determine gaze direction
        # Horizontal: 0.5 is center, <0.4 is left, >0.6 is right
        # Vertical: 0.5 is center, <0.4 is up, >0.6 is down
        
        direction = "center"
        looking_at_screen = True
        
        # Check horizontal gaze
        if avg_h < 0.35 or head_pose['yaw'] < -20:
            direction = "left"
            looking_at_screen = False
        elif avg_h > 0.65 or head_pose['yaw'] > 20:
            direction = "right"
            looking_at_screen = False
        # Check vertical gaze
        elif avg_v < 0.35 or head_pose['pitch'] < -15:
            direction = "up"
            looking_at_screen = avg_v > 0.3  # Allow slight upward glance (thinking)
        elif avg_v > 0.65 or head_pose['pitch'] > 15:
            direction = "down"
            looking_at_screen = False
        
        # Calculate gaze score (0-1, where 1 is looking at screen)
        h_deviation = abs(avg_h - 0.5) * 2  # 0-1 scale
        v_deviation = abs(avg_v - 0.5) * 2
        head_deviation = (abs(head_pose['yaw']) / 45 + abs(head_pose['pitch']) / 30) / 2
        
        gaze_score = 1.0 - ((h_deviation + v_deviation + head_deviation) / 3)
        gaze_score = max(0.0, min(1.0, gaze_score))
        
        return {
            'direction': direction,
            'looking_at_screen': looking_at_screen,
            'gaze_score': gaze_score,
            'iris_h': avg_h,
            'iris_v': avg_v,
            'head_pose': head_pose
        }
    
    def analyze_gaze(self, image: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze gaze with professional accuracy.
        
        Args:
            image: Input BGR image
            timestamp: Current timestamp
            
        Returns:
            Gaze analysis results with violations
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {
                'gaze_detected': False,
                'direction': 'unknown',
                'looking_at_screen': False,
                'gaze_score': 0.0,
                'violations': [],
                'processing_time': (time.time() - start_time) * 1000
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Detect blink
        is_blinking = False
        if self.blink_detection:
            is_blinking = self.detect_blink(landmarks, image.shape)
            self.blink_history.append({
                'is_blink': is_blinking,
                'timestamp': timestamp
            })
        
        # Calculate gaze direction
        gaze_data = self.calculate_gaze_direction(landmarks, image.shape)
        
        # Add to history
        self.gaze_history.append({
            'gaze_score': gaze_data['gaze_score'],
            'looking_at_screen': gaze_data['looking_at_screen'],
            'direction': gaze_data['direction'],
            'timestamp': timestamp
        })
        
        # Check for violations
        violations = []
        
        # Looking away violation
        if len(self.gaze_history) >= 20:  # At least 2 seconds
            recent = list(self.gaze_history)[-50:]
            looking_away_count = sum(1 for g in recent if not g['looking_at_screen'])
            
            if looking_away_count > 40:  # >80% of time
                if self.looking_away_start is None:
                    self.looking_away_start = timestamp
                else:
                    duration = timestamp - self.looking_away_start
                    if duration >= self.looking_away_threshold:
                        if timestamp - self.last_violation > self.violation_cooldown:
                            self.last_violation = timestamp
                            
                            # Determine primary direction
                            directions = [g['direction'] for g in recent if not g['looking_at_screen']]
                            primary_direction = max(set(directions), key=directions.count) if directions else 'away'
                            
                            violations.append({
                                'type': 'looking_away',
                                'direction': primary_direction,
                                'duration': duration,
                                'severity': 'medium',
                                'confidence': 0.90,
                                'message': f'Looking {primary_direction} for {duration:.1f} seconds'
                            })
            else:
                self.looking_away_start = None
        
        # Abnormal blink rate
        if self.blink_detection and len(self.blink_history) >= 50:
            recent_blinks = list(self.blink_history)[-100:]
            blink_count = sum(1 for b in recent_blinks if b['is_blink'])
            blink_rate = blink_count / 10.0  # Blinks per second
            
            # Normal: 15-20 blinks/minute (0.25-0.33 per second)
            # Abnormal: >40 blinks/minute (>0.67 per second)
            if blink_rate > 0.7:
                violations.append({
                    'type': 'abnormal_blink_rate',
                    'blink_rate': blink_rate * 60,  # Convert to per minute
                    'severity': 'low',
                    'confidence': 0.75,
                    'message': f'Abnormal blink rate: {blink_rate * 60:.0f} blinks/minute'
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'gaze_detected': True,
            'direction': gaze_data['direction'],
            'looking_at_screen': gaze_data['looking_at_screen'],
            'gaze_score': gaze_data['gaze_score'],
            'head_pose': gaze_data['head_pose'],
            'iris_position': {
                'horizontal': gaze_data['iris_h'],
                'vertical': gaze_data['iris_v']
            },
            'is_blinking': is_blinking,
            'violations': violations,
            'processing_time': processing_time
        }
    
    def reset(self):
        """Reset all tracking state"""
        self.gaze_history.clear()
        self.blink_history.clear()
        self.looking_away_start = None
        self.last_violation = 0
