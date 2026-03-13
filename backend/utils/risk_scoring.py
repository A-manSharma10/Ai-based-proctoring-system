"""
Risk Scoring Engine for Exam Proctoring
Calculates weighted risk scores from multiple violation types
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    Professional Risk Scoring Engine for Exam Proctoring.
    Implements weighted scoring across all violation categories with temporal persistence.
    """
    
    def __init__(self):
        # Professional weights (sum to 1.0)
        self.weights = {
            'face': 0.25,      # No face / Face mismatch
            'multi_face': 0.20, # Multiple people
            'gaze': 0.15,      # Looking away duration
            'object': 0.25,    # Prohibited objects (phone/book)
            'audio': 0.15      # Speech/Whisper detection
        }
        
        # Base scores for violation types
        self.base_scores = {
            'no_face': 0.8,
            'face_mismatch': 1.0,
            'multiple_faces': 1.0,
            'prohibited_object': 0.9,
            'phone_detected': 1.0,
            'book_detected': 0.7,
            'speech_detected': 0.6,
            'multiple_speakers': 0.9,
            'whispering': 0.7,
            'looking_away': 0.6
        }
    
    def calculate_face_score(self, face_violations: List[Dict[str, Any]]) -> float:
        """Calculate face violation score (missing or mismatch)"""
        if not face_violations: return 0.0
        
        # Max score from any face violation
        scores = []
        for v in face_violations:
            v_type = v.get('type', '').lower()
            if 'no_face' in v_type:
                # Scaled by duration
                duration = v.get('duration', 0)
                scores.append(self.base_scores['no_face'] * min(1.0, duration / 10.0))
            elif 'mismatch' in v_type or 'impersonation' in v_type:
                scores.append(self.base_scores['face_mismatch'])
        
        return max(scores) if scores else 0.0

    def calculate_multi_face_score(self, face_violations: List[Dict[str, Any]]) -> float:
        """Calculate score for multiple faces detected"""
        if not face_violations: return 0.0
        
        multi_face_v = [v for v in face_violations if 'multiple' in v.get('type', '').lower()]
        if not multi_face_v: return 0.0
        
        # Scaled by persistence (duration)
        max_duration = max(v.get('duration', 0) for v in multi_face_v)
        return self.base_scores['multiple_faces'] * min(1.0, max_duration / 5.0)

    def calculate_gaze_score(self, gaze_violations: List[Dict[str, Any]]) -> float:
        """Calculate gaze score based on duration and frequency of looking away"""
        if not gaze_violations: return 0.0
        
        total_duration = sum(v.get('duration', 0) for v in gaze_violations)
        # 1.0 score at 30 seconds total looking away
        return self.base_scores['looking_away'] * min(1.0, total_duration / 30.0)

    def calculate_object_score(self, object_violations: List[Dict[str, Any]]) -> float:
        """Calculate prohibited object score with persistence weighting"""
        if not object_violations: return 0.0
        
        scores = []
        for v in object_violations:
            cat = v.get('category', '').lower()
            obj = v.get('object', '').lower()
            
            # Base score by category
            if 'phone' in cat or 'phone' in obj:
                base = self.base_scores['phone_detected']
            elif 'book' in cat or 'book' in obj:
                base = self.base_scores['book_detected']
            else:
                base = self.base_scores['prohibited_object']
            
            # Weight by persistence (duration seen)
            duration = v.get('duration', 0)
            persistence_weight = min(1.0, duration / 3.0) # Full score after 3s
            
            scores.append(base * persistence_weight)
            
        return max(scores) if scores else 0.0

    def calculate_audio_score(self, audio_violations: List[Dict[str, Any]]) -> float:
        """Calculate audio risk score"""
        if not audio_violations: return 0.0
        
        scores = []
        for v in audio_violations:
            v_type = v.get('type', '').lower()
            if 'multiple' in v_type:
                scores.append(self.base_scores['multiple_speakers'])
            elif 'whisper' in v_type:
                scores.append(self.base_scores['whispering'])
            else:
                scores.append(self.base_scores['speech_detected'])
        
        return max(scores) if scores else 0.0

    def calculate_risk_score(self, 
                            violations: Dict[str, List[Dict[str, Any]]],
                            exam_duration_minutes: float = 0) -> Dict[str, Any]:
        """
        Calculate professional risk score.
        Weighted sum: w1*Face + w2*MultiFace + w3*Gaze + w4*Object + w5*Audio
        """
        try:
            # Face component
            face_score = self.calculate_face_score(violations.get('face', []))
            
            # Multi-face component (separated for accuracy)
            multi_face_score = self.calculate_multi_face_score(violations.get('face', []))
            
            # Gaze component
            gaze_score = self.calculate_gaze_score(violations.get('gaze', []))
            
            # Object component
            object_score = self.calculate_object_score(violations.get('object', []))
            
            # Audio component
            audio_score = self.calculate_audio_score(violations.get('audio', []))
            
            # Weighted calculation
            total_risk = (
                face_score * self.weights['face'] +
                multi_face_score * self.weights['multi_face'] +
                gaze_score * self.weights['gaze'] +
                object_score * self.weights['object'] +
                audio_score * self.weights['audio']
            )
            
            # Scale to 0-100
            total_score = min(100.0, total_risk * 100.0)
            
            # Determine severity
            if total_score < 25: severity = 'Low'
            elif total_score < 50: severity = 'Medium'
            elif total_score < 75: severity = 'High'
            else: severity = 'Critical'
            
            color = 'green' if total_score < 25 else 'yellow' if total_score < 50 else 'orange' if total_score < 75 else 'red'
            
            return {
                'total_score': round(total_score, 2),
                'breakdown': {
                    'face_score': round(face_score * 100, 2),
                    'multi_face_score': round(multi_face_score * 100, 2),
                    'gaze_score': round(gaze_score * 100, 2),
                    'object_score': round(object_score * 100, 2),
                    'audio_score': round(audio_score * 100, 2)
                },
                'severity': severity,
                'color': color,
                'explanation': self.generate_explanation(total_score, violations)
            }
        except Exception as e:
            logger.error(f"Professional risk calculation failed: {e}")
            return {'total_score': 0, 'severity': 'Low', 'error': str(e)}

    def generate_explanation(self, score: float, violations: Dict) -> str:
        """Generate explainable breakdown of the risk score"""
        reasons = []
        if violations.get('face'):
            if any('multiple' in v.get('type','') for v in violations['face']):
                reasons.append("Multiple people detected in frame")
            if any('no' in v.get('type','') for v in violations['face']):
                reasons.append("Subject missing from camera view")
                
        if violations.get('object'):
            reasons.append(f"Prohibited objects ({len(violations['object'])}) detected")
            
        if violations.get('gaze'):
            reasons.append("Frequent or prolonged gaze away from screen")
            
        if violations.get('audio'):
            reasons.append("Suspected human speech or multiple speakers detected")
            
        if not reasons:
            return "No significant suspicious behavior detected."
            
        return f"Risk factor breakdown: {', '.join(reasons)}. Overall integrity score: {100-score:.1f}%."

