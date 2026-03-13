import numpy as np
import mediapipe as mp
import cv2
import time
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ProfessionalBehavioralAnalyzer:
    """
    Professional Behavioral Analysis System for Exam Proctoring.
    Tracks body pose, hand gestures, and temporal movement patterns.
    """
    
    def __init__(self, 
                 history_size: int = 30,
                 movement_threshold: float = 0.05,
                 sudden_move_threshold: float = 0.15):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 1 is balanced, 2 is heavy
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Temporal memory
        self.history_size = history_size
        self.pose_history = deque(maxlen=history_size)
        self.hand_history = deque(maxlen=history_size)
        
        # Thresholds
        self.movement_threshold = movement_threshold
        self.sudden_move_threshold = sudden_move_threshold
        
    def analyze_frame(self, image: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Main analysis pipeline for a single frame"""
        start_time = time.time()
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Body Pose Analysis
        pose_results = self.pose.process(rgb_image)
        pose_data = self._extract_pose_data(pose_results, w, h)
        self.pose_history.append({'data': pose_data, 'ts': timestamp})
        
        # 2. Hand Gesture Analysis
        hand_results = self.hands.process(rgb_image)
        hand_data = self._extract_hand_data(hand_results, w, h)
        self.hand_history.append({'data': hand_data, 'ts': timestamp})
        
        # 3. Temporal Movement Analysis
        movement_stats = self._analyze_temporal_movement()
        
        # 4. Generate Violations
        violations = self._detect_violations(pose_data, hand_data, movement_stats)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'pose': pose_data,
            'hands': hand_data,
            'movement': movement_stats,
            'violations': violations,
            'risk_score': self._calculate_risk(violations, movement_stats),
            'processing_time': processing_time
        }

    def _extract_pose_data(self, results, w, h) -> Dict[str, Any]:
        """Extract key pose landmarks and calculate angles"""
        if not results.pose_landmarks:
            return {'detected': False}
            
        landmarks = results.pose_landmarks.landmark
        
        # Get key points
        nose = np.array([landmarks[0].x, landmarks[0].y])
        l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        l_elbow = np.array([landmarks[13].x, landmarks[13].y])
        r_elbow = np.array([landmarks[14].x, landmarks[14].y])
        l_wrist = np.array([landmarks[15].x, landmarks[15].y])
        r_wrist = np.array([landmarks[16].x, landmarks[16].y])
        
        # Calculate features
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        
        # Lean factors
        lean_x = nose[0] - shoulder_mid[0]
        
        # Hand to face detection
        l_hand_to_face = np.linalg.norm(l_wrist - nose)
        r_hand_to_face = np.linalg.norm(r_wrist - nose)
        
        return {
            'detected': True,
            'nose': nose.tolist(),
            'mid_shoulder': shoulder_mid.tolist(),
            'lean_x': float(lean_x),
            'shoulder_width': float(shoulder_width),
            'hands_near_face': bool(l_hand_to_face < 0.15 or r_hand_to_face < 0.15),
            'elbows_out': bool(np.linalg.norm(l_elbow[0] - l_shoulder[0]) > 0.2 or 
                              np.linalg.norm(r_elbow[0] - r_shoulder[0]) > 0.2)
        }

    def _extract_hand_data(self, results, w, h) -> Dict[str, Any]:
        """Extract hand positions and specific gestures"""
        if not results.multi_hand_landmarks:
            return {'count': 0, 'hands': []}
            
        hand_data = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            wrist = np.array([landmarks[0].x, landmarks[0].y])
            fingers_up = 0
            # Simple finger counting
            for tip_idx in [8, 12, 16, 20]:
                if landmarks[tip_idx].y < landmarks[tip_idx-2].y:
                    fingers_up += 1
            
            hand_data.append({
                'wrist': wrist.tolist(),
                'fingers_up': fingers_up,
                'is_left': results.multi_handedness[idx].classification[0].label == 'Left'
            })
            
        return {'count': len(hand_data), 'hands': hand_data}

    def _analyze_temporal_movement(self) -> Dict[str, Any]:
        """Analyze stability and suddenness of movement"""
        if len(self.pose_history) < 5:
            return {'stability': 1.0, 'sudden_movement': False, 'vibrancy': 0}
            
        valid_poses = [p['data'] for p in self.pose_history if p['data']['detected']]
        if len(valid_poses) < 2:
            return {'stability': 1.0, 'sudden_movement': False, 'vibrancy': 0}
            
        # Extract nose positions for stability track
        noses = np.array([p['nose'] for p in valid_poses])
        diffs = np.linalg.norm(np.diff(noses, axis=0), axis=1)
        
        max_diff = np.max(diffs)
        avg_diff = np.mean(diffs)
        
        return {
            'stability': float(max(0, 1.0 - (avg_diff * 10))),
            'sudden_movement': bool(max_diff > self.sudden_move_threshold),
            'vibrancy': float(avg_diff)
        }

    def _detect_violations(self, pose, hands, movement) -> List[Dict[str, Any]]:
        """Identify suspicious behavioral patterns"""
        violations = []
        
        # 1. Unusual Position / Leaning
        if pose.get('detected'):
            if abs(pose['lean_x']) > 0.15:
                violations.append({
                    'type': 'excessive_lean',
                    'severity': 'medium',
                    'message': 'Student is leaning significantly out of center',
                    'confidence': 0.8
                })
            
            if pose['hands_near_face'] and not movement['sudden_movement']:
                violations.append({
                    'type': 'hand_to_face',
                    'severity': 'low',
                    'message': 'Hands detected near face (possible whispering or covering mouth)',
                    'confidence': 0.7
                })
        
        # 2. Hand Activity
        if hands['count'] > 2:
            violations.append({
                'type': 'unauthorized_hands',
                'severity': 'high',
                'message': f'Multiple hands ({hands["count"]}) detected in frame',
                'confidence': 0.95
            })
            
        # 3. Sudden Movement
        if movement['sudden_movement']:
            violations.append({
                'type': 'sudden_movement',
                'severity': 'medium',
                'message': 'Sudden jerky movements detected',
                'confidence': 0.85
            })
            
        return violations

    def _calculate_risk(self, violations, movement) -> float:
        """Calculate weighted behavioral risk score (0-1)"""
        if not violations:
            # Baseline risk from low stability
            return max(0, 0.1 * (1.0 - movement['stability']))
            
        score = 0
        weights = {'high': 0.6, 'medium': 0.3, 'low': 0.1}
        
        for v in violations:
            score += weights.get(v['severity'], 0.1)
            
        return min(1.0, score)
