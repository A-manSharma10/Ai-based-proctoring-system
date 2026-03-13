from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
import base64
import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from professional_behavioral import ProfessionalBehavioralAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Professional Behavioral Analysis Service", version="2.0.0")

# Session-based professional analyzers
session_analyzers: Dict[int, ProfessionalBehavioralAnalyzer] = {}

# Pydantic models
class BehavioralAnalysisRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: int

class BehavioralPattern(BaseModel):
    pattern_type: str
    confidence: float
    message: str
    severity: str  # low, medium, high, critical

class BehavioralAnalysisResponse(BaseModel):
    suspicious_behavior: bool
    risk_score: float
    detected_behaviors: List[BehavioralPattern]
    processing_time: float
    movement_stats: Dict[str, Any]

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

@app.post("/analyze", response_model=BehavioralAnalysisResponse)
async def analyze_behavior(request: BehavioralAnalysisRequest):
    """Analyze behavioral patterns with professional-grade accuracy"""
    try:
        start_time = time.time()
        session_id = request.session_id
        
        # Get or create professional analyzer for this session
        if session_id not in session_analyzers:
            session_analyzers[session_id] = ProfessionalBehavioralAnalyzer(
                history_size=30,
                movement_threshold=0.05
            )
        
        analyzer = session_analyzers[session_id]
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Run professional analysis pipeline
        results = analyzer.analyze_frame(image, start_time)
        
        # Map violations to BehavioralPattern models
        detected_behaviors = []
        for v in results['violations']:
            detected_behaviors.append(BehavioralPattern(
                pattern_type=v['type'],
                confidence=v['confidence'],
                message=v['message'],
                severity=v['severity']
            ))
            
        return BehavioralAnalysisResponse(
            suspicious_behavior=len(detected_behaviors) > 0,
            risk_score=results['risk_score'],
            detected_behaviors=detected_behaviors,
            processing_time=results['processing_time'],
            movement_stats=results['movement']
        )

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Behavioral analysis error: {e}")
        raise HTTPException(status_code=500, detail="Behavioral analysis failed")

@app.get("/session-history/{session_id}")
async def get_session_history(session_id: int):
    """Get behavioral history for a session"""
    try:
        if session_id not in behavioral_history:
            return {"session_id": session_id, "history": [], "message": "No history found"}
        
        history = list(behavioral_history[session_id])
        
        return {
            "session_id": session_id,
            "history_length": len(history),
            "history": history[-10:],  # Return last 10 entries
            "summary": {
                "avg_risk_score": np.mean([
                    calculate_risk_score(
                        frame.get("body_pose_analysis", {}),
                        frame.get("hand_analysis", {}),
                        {"movement_frequency": 0, "position_stability": 1, "repetitive_patterns": [], "sudden_movements": False}
                    ) for frame in history
                ]) if history else 0
            }
        }
    except Exception as e:
        logger.error(f"Get session history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session history")

@app.delete("/session-history/{session_id}")
async def clear_session_history(session_id: int):
    """Clear behavioral history for a session"""
    try:
        if session_id in behavioral_history:
            del behavioral_history[session_id]
            return {"message": f"History cleared for session {session_id}"}
        else:
            return {"message": f"No history found for session {session_id}"}
    except Exception as e:
        logger.error(f"Clear session history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session history")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "behavioral_analysis"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)