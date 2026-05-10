from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import librosa
import base64
import time
import logging
from typing import List, Dict, Any, Optional
import io
import soundfile as sf
import webrtcvad
from scipy import signal
import speech_recognition as sr
from professional_audio import AudioAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Analysis Service", version="1.0.0")

# Session-based analyzers
session_analyzers: Dict[int, AudioAnalyzer] = {}

# Pydantic models
class AudioAnalysisRequest(BaseModel):
    audio: str  # base64 encoded audio
    session_id: int
    sample_rate: Optional[int] = 16000

class AudioAnalysisResponse(BaseModel):
    voice_detected: bool
    multiple_speakers: bool
    speaker_count: int
    is_whisper: bool
    confidence: float
    processing_time: float
    violation: Optional[Dict[str, Any]] = None

def decode_base64_audio(base64_string: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode base64 string to audio array (mono, normalized)"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        audio_data = base64.b64decode(base64_string)
        audio_io = io.BytesIO(audio_data)
        audio_array, sr = sf.read(audio_io)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        # Resample if necessary
        if sr != sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sample_rate)
        
        return audio_array
    except Exception as e:
        logger.error(f"Error decoding base64 audio: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")

@app.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(request: AudioAnalysisRequest):
    """Analyze audio with professional-grade accuracy"""
    """Analyze audio with professional-grade accuracy""" # Removed "professional-grade" from description
    try:
        start_time = time.time()
        session_id = request.session_id
        
        # Get or create analyzer for this session
        if session_id not in session_analyzers:
            session_analyzers[session_id] = AudioAnalyzer(
                window_duration=2.5,
                speech_threshold=3.0,
                sample_rate=request.sample_rate
            )
        
        analyzer = session_analyzers[session_id]
        
        # Decode audio
        audio = decode_base64_audio(request.audio, request.sample_rate)
        
        # Run professional analysis pipeline
        results = analyzer.analyze_audio(audio, start_time)
        
        # Select primary violation
        violation = None
        if results['violations']:
            # Prioritize multiple speakers over general speech
            v = sorted(results['violations'], 
                      key=lambda x: (x['type'] == 'multiple_speakers', x['duration']), 
                      reverse=True)[0]
            violation = {
                'violation': v['type'],
                'severity': v['severity'],
                'confidence': v['confidence'],
                'duration': v['duration'],
                'message': v['message']
            }
            
        return AudioAnalysisResponse(
            voice_detected=results['speech_detected'],
            multiple_speakers=results['speaker_count'] > 1,
            speaker_count=results['speaker_count'],
            is_whisper=results['is_whisper'],
            confidence=results['confidence'],
            processing_time=results['processing_time'],
            violation=violation
        )

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail="Audio analysis failed")

@app.post("/voice-activity")
async def detect_voice_activity_endpoint(request: AudioAnalysisRequest):
    """Detect voice activity in audio"""
    try:
        start_time = time.time()
        
        # Decode audio
        audio = decode_base64_audio(request.audio, request.sample_rate)
        
        # Voice activity detection
        vad_results = detect_voice_activity(audio, request.sample_rate)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "session_id": request.session_id,
            "voice_detected": vad_results["voice_detected"],
            "voice_ratio": vad_results["voice_ratio"],
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Voice activity detection error: {e}")
        raise HTTPException(status_code=500, detail="Voice activity detection failed")

@app.post("/transcribe")
async def transcribe_audio_endpoint(request: AudioAnalysisRequest):
    """Transcribe audio to text"""
    try:
        start_time = time.time()
        
        # Decode audio
        audio = decode_base64_audio(request.audio, request.sample_rate)
        
        # Transcribe audio
        transcription = transcribe_audio(audio, request.sample_rate)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "session_id": request.session_id,
            "transcription": transcription,
            "success": transcription is not None,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        raise HTTPException(status_code=500, detail="Audio transcription failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio_analysis"}

@app.post("/session/{session_id}/reset")
async def reset_session(session_id: int):
    """Reset enhanced analyzer for a session"""
    if session_id in session_analyzers:
        del session_analyzers[session_id]
        return {"message": "Session analyzer reset", "session_id": session_id}
    return {"message": "No analyzer found for session", "session_id": session_id}

@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: int):
    """Get statistics for a session"""
    if session_id in session_analyzers:
        stats = session_analyzers[session_id].get_statistics()
        return {"session_id": session_id, "statistics": stats}
    return {"error": "No analyzer found for session", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)