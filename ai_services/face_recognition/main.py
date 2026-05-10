from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import base64
import time
import logging
from typing import List, Optional, Dict, Any
from professional_face_detector import FaceDetector
from professional_gaze_tracker import GazeTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Service", version="1.0.0")

# Session-based detectors
session_face_detectors: Dict[int, FaceDetector] = {}
session_gaze_trackers: Dict[int, GazeTracker] = {}

# Pydantic models
class FaceRegistrationRequest(BaseModel):
    image: str  # base64 encoded image
    user_id: str

class FaceVerificationRequest(BaseModel):
    image: str  # base64 encoded image
    user_id: int

class FaceAnalysisRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: int

class FaceAnalysisResponse(BaseModel):
    face_detected: bool
    face_count: int
    multiple_faces: bool
    attention_score: float
    gaze_direction: str
    head_pose: Dict[str, float]
    confidence: float
    processing_time: float
    violation: Optional[Dict[str, Any]] = None
    blink_detected: bool = False
    iris_position: Optional[Dict[str, float]] = None

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image (BGR format)"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.post("/register-face")
async def register_face(request: FaceRegistrationRequest):
    """Register a face for a user"""
    try:
        start_time = time.time()
        image = decode_base64_image(request.image)
        
        # Use face_recognition for embeddings (deep learning based)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(face_encodings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please use image with single face")
        
        face_encoding = face_encodings[0]
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "message": "Face registered successfully",
            "embedding": face_encoding.tolist(),
            "processing_time": processing_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        raise HTTPException(status_code=500, detail="Face registration failed")

@app.post("/verify-face")
async def verify_face(request: FaceVerificationRequest):
    """Verify a face against stored encoding with professional accuracy"""
    try:
        start_time = time.time()
        image = decode_base64_image(request.image)
        
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "No face detected",
                "processing_time": (time.time() - start_time) * 1000
            }
        
        face_encoding = face_encodings[0]
        stored_encoding = get_face_encoding(request.user_id)
        
        if stored_encoding is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "Face profile not found",
                "processing_time": (time.time() - start_time) * 1000
            }
            
        stored_encoding_np = np.array(stored_encoding)
        face_distances = face_recognition.face_distance([stored_encoding_np], face_encoding)
        
        # Professional threshold: 0.5 (more strict than 0.6)
        verified = bool(face_distances[0] < 0.5)
        confidence = float(1.0 - face_distances[0])
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "verified": verified,
            "confidence": confidence,
            "message": "Verification passed" if verified else "Face mismatch detected",
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        raise HTTPException(status_code=500, detail="Face verification failed")

@app.post("/analyze", response_model=FaceAnalysisResponse)
async def analyze_face(request: FaceAnalysisRequest):
    """Analyze face presence and gaze direction"""
    try:
        start_time = time.time()
        session_id = request.session_id
        
        # Initialize detectors for session
        if session_id not in session_face_detectors:
            session_face_detectors[session_id] = FaceDetector(
                no_face_threshold=3.0,
                multiple_face_threshold=1.0
            )
        if session_id not in session_gaze_trackers:
            session_gaze_trackers[session_id] = GazeTracker(
                looking_away_threshold=2.0
            )
            
        face_detector = session_face_detectors[session_id]
        gaze_tracker = session_gaze_trackers[session_id]
        
        image = decode_base64_image(request.image)
        
        # 1. Run Professional Face Detection & Tracking
        face_results = face_detector.detect_and_track(image, start_time)
        
        # 2. Run Professional Gaze & Pupil Tracking
        gaze_results = gaze_tracker.analyze_gaze(image, start_time)
        
        # Select primary violation
        violations = face_results['violations'] + gaze_results['violations']
        primary_violation = None
        if violations:
            primary_violation = sorted(violations, 
                                      key=lambda x: (x['severity'] == 'critical', x['confidence']), 
                                      reverse=True)[0]
            
        return FaceAnalysisResponse(
            face_detected=face_results['face_detected'],
            face_count=face_results['face_count'],
            multiple_faces=face_results['face_count'] > 1,
            attention_score=gaze_results['gaze_score'],
            gaze_direction=gaze_results['direction'],
            head_pose=gaze_results['head_pose'],
            confidence=face_results['confidence'],
            processing_time=face_results['processing_time'] + gaze_results['processing_time'],
            violation=primary_violation,
            blink_detected=gaze_results.get('is_blinking', False),
            iris_position=gaze_results.get('iris_position')
        )

        
    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        raise HTTPException(status_code=500, detail="Face analysis failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "face_recognition"}

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: int):
    """Reset temporal and gaze trackers for a session"""
    cleared = []
    if session_id in session_trackers:
        del session_trackers[session_id]
        cleared.append("face_tracker")
    if session_id in gaze_trackers:
        del gaze_trackers[session_id]
        cleared.append("gaze_tracker")
    
    if cleared:
        return {"message": "Session trackers reset", "session_id": session_id, "cleared": cleared}
    return {"message": "No trackers found for session", "session_id": session_id}

@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: int):
    """Get statistics for a session"""
    stats = {}
    
    if session_id in session_trackers:
        stats['face_tracking'] = session_trackers[session_id].get_statistics()
    
    if session_id in gaze_trackers:
        stats['gaze_tracking'] = gaze_trackers[session_id].get_statistics()
    
    if stats:
        return {"session_id": session_id, "statistics": stats}
    return {"error": "No trackers found for session", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)