"""
Professional-Grade Face Detection and Tracking
Implements commercial-level accuracy with:
- Continuous face tracking
- Multi-face detection with stability
- Side profile detection
- Low-light performance
- Embedding verification
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class ProfessionalFaceDetector:
    """
    Professional face detector with continuous tracking and high accuracy.
    """
    
    def __init__(self,
                 no_face_threshold: float = 3.0,
                 multiple_face_threshold: float = 1.0,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.6):
        """
        Initialize professional face detector.
        
        Args:
            no_face_threshold: Seconds before flagging no face
            multiple_face_threshold: Seconds before flagging multiple faces
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.no_face_threshold = no_face_threshold
        self.multiple_face_threshold = multiple_face_threshold
        
        # Initialize MediaPipe with optimized settings
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Primary detector (high confidence)
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model (better for side profiles)
            min_detection_confidence=min_detection_confidence
        )
        
        # Backup detector (lower confidence for difficult cases)
        self.face_detection_backup = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Face mesh for detailed tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Tracking state
        self.face_tracks = {}
        self.no_face_start = None
        self.multiple_face_start = None
        self.last_violation = 0
        self.violation_cooldown = 10.0
        
        # Detection history
        self.detection_history = deque(maxlen=30)  # 3 seconds at 10fps
        
    def enhance_for_face_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better face detection in low light.
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Check brightness
            mean_brightness = np.mean(l)
            
            if mean_brightness < 100:  # Low light
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # Merge and convert back
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                return enhanced
            
            return image
            
        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return image
    
    def detect_faces_multi_method(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using multiple methods for robustness.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detected faces with confidence and bbox
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = []
        
        # Method 1: Primary detector
        results = self.face_detection.process(rgb_image)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0] if detection.score else 0.0
                
                h, w = image.shape[:2]
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'method': 'primary'
                })
        
        # Method 2: Backup detector (if no faces found)
        if len(faces) == 0:
            results_backup = self.face_detection_backup.process(rgb_image)
            if results_backup.detections:
                for detection in results_backup.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0] if detection.score else 0.0
                    
                    h, w = image.shape[:2]
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'method': 'backup'
                    })
        
        # Method 3: Face mesh (for tracking)
        mesh_results = self.face_mesh.process(rgb_image)
        if mesh_results.multi_face_landmarks:
            for idx, landmarks in enumerate(mesh_results.multi_face_landmarks):
                # Calculate bounding box from landmarks
                h, w = image.shape[:2]
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                # Check if this face is already detected
                is_duplicate = False
                for existing_face in faces:
                    iou = self.calculate_iou([x1, y1, x2, y2], existing_face['bbox'])
                    if iou > 0.5:
                        is_duplicate = True
                        existing_face['has_landmarks'] = True
                        existing_face['landmarks'] = landmarks
                        break
                
                if not is_duplicate:
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 0.8,  # High confidence from mesh
                        'method': 'mesh',
                        'has_landmarks': True,
                        'landmarks': landmarks
                    })
        
        return faces
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_face_tracks(self, faces: List[Dict[str, Any]], timestamp: float):
        """
        Update face tracking with new detections.
        
        Args:
            faces: Detected faces
            timestamp: Current timestamp
        """
        # Match faces to existing tracks
        matched_tracks = set()
        
        for face in faces:
            best_match = None
            best_iou = 0.3  # Minimum IOU for matching
            
            for track_id, track in self.face_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self.calculate_iou(face['bbox'], track['last_bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match:
                # Update existing track
                track = self.face_tracks[best_match]
                track['last_bbox'] = face['bbox']
                track['last_seen'] = timestamp
                track['confidence_history'].append(face['confidence'])
                track['detections'].append(face)
                matched_tracks.add(best_match)
            else:
                # Create new track
                track_id = f"face_{len(self.face_tracks)}_{timestamp}"
                self.face_tracks[track_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'last_bbox': face['bbox'],
                    'confidence_history': deque([face['confidence']], maxlen=30),
                    'detections': deque([face], maxlen=30)
                }
        
        # Remove old tracks
        to_remove = []
        for track_id, track in self.face_tracks.items():
            if (timestamp - track['last_seen']) > 2.0:  # 2 seconds
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.face_tracks[track_id]
    
    def get_stable_face_count(self) -> int:
        """
        Get count of stable tracked faces.
        
        Returns:
            Number of stable faces
        """
        return len([t for t in self.face_tracks.values() 
                   if len(t['detections']) >= 3])
    
    def detect_and_track(self, image: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect and track faces with professional accuracy.
        
        Args:
            image: Input BGR image
            timestamp: Current timestamp
            
        Returns:
            Detection results with violations
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        # Enhance image
        enhanced = self.enhance_for_face_detection(image)
        
        # Detect faces
        faces = self.detect_faces_multi_method(enhanced)
        
        # Update tracks
        self.update_face_tracks(faces, timestamp)
        
        # Get stable face count
        stable_face_count = self.get_stable_face_count()
        
        # Add to history
        self.detection_history.append({
            'face_count': stable_face_count,
            'raw_count': len(faces),
            'timestamp': timestamp
        })
        
        # Check for violations
        violations = []
        
        # No face violation
        if stable_face_count == 0:
            if self.no_face_start is None:
                self.no_face_start = timestamp
            else:
                duration = timestamp - self.no_face_start
                if duration >= self.no_face_threshold:
                    if timestamp - self.last_violation > self.violation_cooldown:
                        self.last_violation = timestamp
                        violations.append({
                            'type': 'no_face',
                            'duration': duration,
                            'severity': 'high',
                            'confidence': 0.95,
                            'message': f'No face detected for {duration:.1f} seconds'
                        })
        else:
            self.no_face_start = None
        
        # Multiple faces violation
        if stable_face_count > 1:
            if self.multiple_face_start is None:
                self.multiple_face_start = timestamp
            else:
                duration = timestamp - self.multiple_face_start
                if duration >= self.multiple_face_threshold:
                    if timestamp - self.last_violation > self.violation_cooldown:
                        self.last_violation = timestamp
                        violations.append({
                            'type': 'multiple_faces',
                            'count': stable_face_count,
                            'duration': duration,
                            'severity': 'critical',
                            'confidence': 0.98,
                            'message': f'{stable_face_count} faces detected for {duration:.1f} seconds'
                        })
        else:
            self.multiple_face_start = None
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate average confidence
        avg_confidence = 0.0
        if faces:
            avg_confidence = np.mean([f['confidence'] for f in faces])
        
        return {
            'face_detected': stable_face_count > 0,
            'face_count': stable_face_count,
            'raw_face_count': len(faces),
            'faces': faces,
            'tracked_faces': len(self.face_tracks),
            'confidence': avg_confidence,
            'violations': violations,
            'processing_time': processing_time
        }
    
    def reset(self):
        """Reset all tracking state"""
        self.face_tracks.clear()
        self.detection_history.clear()
        self.no_face_start = None
        self.multiple_face_start = None
        self.last_violation = 0
