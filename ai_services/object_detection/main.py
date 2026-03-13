from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import logging
from typing import List, Dict, Any, Optional
import os
from professional_detector import ProfessionalObjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Professional Object Detection Service", version="2.0.0")

# Initialize YOLO model
try:
    # Use YOLOv8 small model for better accuracy than nano, but still fast
    model = YOLO('yolov8s.pt') 
    logger.info("Professional YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

# Session-based professional detectors
session_detectors: Dict[int, ProfessionalObjectDetector] = {}

# Pydantic models
class ObjectDetectionRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: int

class DetectedObject(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    is_prohibited: bool
    category: Optional[str] = None
    duration: Optional[float] = None

class ObjectDetectionResponse(BaseModel):
    objects_detected: List[DetectedObject]
    prohibited_objects: List[DetectedObject]
    person_count: int
    multiple_people: bool
    processing_time: float
    violation: Optional[Dict[str, Any]] = None
    small_objects_detected: int

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

@app.post("/detect", response_model=ObjectDetectionResponse)
async def detect_objects(request: ObjectDetectionRequest):
    """Detect objects with professional-grade accuracy"""
    try:
        start_time = time.time()
        session_id = request.session_id
        
        if model is None:
            raise HTTPException(status_code=503, detail="YOLO model not available")
        
        # Get or create professional detector for this session
        if session_id not in session_detectors:
            session_detectors[session_id] = ProfessionalObjectDetector(
                model=model,
                input_size=960,  # High resolution for small objects
                frame_buffer_size=15,
                min_frames_for_detection=8,
                confidence_threshold=0.4, # Lower for better recall on small objects
                persistence_threshold=1.5
            )
        
        detector = session_detectors[session_id]
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Run professional detection pipeline
        results = detector.detect_frame(image, start_time)
        
        # Convert to response format
        detected_objects = []
        for det in results['detections']:
            is_proh, _ = detector.is_prohibited(det['class_name'])
            detected_objects.append(DetectedObject(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                is_prohibited=is_proh,
                duration=det.get('duration')
            ))
            
        prohibited_objects = []
        for det in results['prohibited_objects']:
            prohibited_objects.append(DetectedObject(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                is_prohibited=True,
                category=det['category'],
                duration=det.get('duration')
            ))
            
        # Select primary violation if any
        violation = None
        if results['violations']:
            # Sort by severity and duration
            v = sorted(results['violations'], 
                      key=lambda x: (x['severity'] == 'critical', x['duration']), 
                      reverse=True)[0]
            violation = {
                'violation': v['type'],
                'object_type': v.get('object', v.get('category')),
                'confidence': v['confidence'],
                'duration': v['duration'],
                'severity': v['severity'],
                'message': v['message']
            }
            
        return ObjectDetectionResponse(
            objects_detected=detected_objects,
            prohibited_objects=prohibited_objects,
            person_count=results['person_count'],
            multiple_people=results['person_count'] > 1,
            processing_time=results['processing_time'],
            violation=violation,
            small_objects_detected=results['small_objects_detected']
        )

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        raise HTTPException(status_code=500, detail="Object detection failed")

@app.post("/detect-custom")
async def detect_custom_objects(request: ObjectDetectionRequest):
    """Detect objects with custom prohibited items list"""
    try:
        start_time = time.time()
        
        if model is None:
            raise HTTPException(status_code=503, detail="YOLO model not available")
        
        # Decode image
        image = decode_base64_image(request.image)
        
        # Run YOLO detection with lower confidence for more detections
        results = model(image, conf=0.3, verbose=False)
        
        # Process results with custom logic
        detected_objects = []
        prohibited_count = 0
        person_count = 0
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    
                    if class_name.lower() == 'person':
                        person_count += 1
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    is_prohibited = is_prohibited_object(class_name)
                    if is_prohibited:
                        prohibited_count += 1
                    
                    detected_objects.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "prohibited": is_prohibited
                    })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "session_id": request.session_id,
            "total_objects": len(detected_objects),
            "prohibited_objects": [obj for obj in detected_objects if obj["prohibited"]],
            "person_count": person_count,
            "multiple_people": person_count > 1,
            "processing_time": processing_time,
            "all_detections": detected_objects
        }
        
    except Exception as e:
        logger.error(f"Custom object detection error: {e}")
        raise HTTPException(status_code=500, detail="Custom object detection failed")

@app.get("/prohibited-objects")
async def get_prohibited_objects():
    """Get list of prohibited objects"""
    return {
        "prohibited_objects": list(PROHIBITED_OBJECTS),
        "description": "Objects that are not allowed during exam sessions"
    }

@app.post("/update-prohibited-objects")
async def update_prohibited_objects(objects: List[str]):
    """Update list of prohibited objects"""
    try:
        global PROHIBITED_OBJECTS
        PROHIBITED_OBJECTS.update(obj.lower() for obj in objects)
        
        return {
            "message": "Prohibited objects updated successfully",
            "prohibited_objects": list(PROHIBITED_OBJECTS)
        }
    except Exception as e:
        logger.error(f"Error updating prohibited objects: {e}")
        raise HTTPException(status_code=500, detail="Failed to update prohibited objects")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "model_type": "YOLOv8",
        "model_size": "nano",
        "classes": len(model.names) if hasattr(model, 'names') else 0,
        "status": "loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "healthy" if model is not None else "unhealthy"
    return {
        "status": model_status,
        "service": "object_detection",
        "model_loaded": model is not None
    }

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: int):
    """Reset enhanced detector for a session"""
    if session_id in session_detectors:
        del session_detectors[session_id]
        return {"message": "Session detector reset", "session_id": session_id}
    return {"message": "No detector found for session", "session_id": session_id}

@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: int):
    """Get statistics for a session"""
    if session_id in session_detectors:
        stats = session_detectors[session_id].get_statistics()
        return {"session_id": session_id, "statistics": stats}
    return {"error": "No detector found for session", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)