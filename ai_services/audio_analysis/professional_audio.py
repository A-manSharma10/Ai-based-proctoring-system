"""
Audio Analysis Module
Handles speech detection, speaker count estimation, and noise filtering for proctoring.
"""

import numpy as np
import librosa
from collections import deque
from typing import Dict, Any, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Audio analyzer for speech presence and speaker count identification.
    """
    
    def __init__(self,
                 window_duration: float = 2.5,
                 speech_threshold: float = 4.0,
                 noise_baseline_samples: int = 15,
                 sample_rate: int = 16000):
        """
        Initialize audio analyzer parameters.
        
        Args:
            window_duration: Duration of analysis window in seconds
            speech_threshold: Seconds of speech before flagging
            noise_baseline_samples: Samples to establish noise baseline
            sample_rate: Audio sample rate
        """
        self.window_duration = window_duration
        self.speech_threshold = speech_threshold
        self.noise_baseline_samples = noise_baseline_samples
        self.sample_rate = sample_rate
        
        # Tracking state
        self.speech_history = deque(maxlen=50)  # 5 seconds at 10 chunks/s
        self.noise_baseline = None
        self.noise_samples = []
        self.speech_start = None
        self.last_violation = 0
        self.violation_cooldown = 10.0
        
        # Noise profiles for filtering
        self.noise_profiles = {
            'fan': {'freq_range': (50, 200), 'energy_threshold': 0.001},
            'keyboard': {'freq_range': (2000, 8000), 'energy_threshold': 0.0005},
            'ambient': {'freq_range': (20, 500), 'energy_threshold': 0.002}
        }
        
    def establish_noise_baseline(self, audio: np.ndarray):
        """
        Establish baseline noise level from initial samples.
        
        Args:
            audio: Audio signal
        """
        if len(self.noise_samples) < self.noise_baseline_samples:
            energy = np.sum(audio ** 2) / len(audio)
            self.noise_samples.append(energy)
            
            if len(self.noise_samples) == self.noise_baseline_samples:
                self.noise_baseline = np.mean(self.noise_samples)
                logger.info(f"Audio noise baseline established: {self.noise_baseline:.6f}")
    
    def filter_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter out common noise types.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Filtered audio
        """
        try:
            # Compute spectrogram
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Create noise mask
            noise_mask = np.ones_like(magnitude)
            
            for noise_type, profile in self.noise_profiles.items():
                freq_min, freq_max = profile['freq_range']
                freq_indices = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
                
                # Reduce magnitude in noise frequency range
                for idx in freq_indices:
                    if np.mean(magnitude[idx, :]) < profile['energy_threshold']:
                        noise_mask[idx, :] *= 0.3  # Attenuate
            
            # Apply mask
            filtered_magnitude = magnitude * noise_mask
            
            # Reconstruct audio
            filtered_D = filtered_magnitude * np.exp(1j * phase)
            filtered_audio = librosa.istft(filtered_D)
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Noise filtering failed: {e}")
            return audio
    
    def detect_speech_advanced(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Advanced speech detection using multiple features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Speech detection results
        """
        try:
            # Calculate features
            # 1. Zero Crossing Rate (speech has moderate ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            mean_zcr = np.mean(zcr)
            
            # 2. Spectral Centroid (speech has specific frequency range)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            # 3. MFCCs (speech has characteristic MFCC pattern)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # 4. Energy
            energy = np.sum(audio ** 2) / len(audio)
            
            # 5. Spectral Rolloff (speech has specific rolloff)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            mean_rolloff = np.mean(spectral_rolloff)
            
            # Speech detection logic
            is_speech = False
            confidence = 0.0
            
            # Check if above noise baseline
            if self.noise_baseline and energy > self.noise_baseline * 2.0:
                # Speech characteristics:
                # - ZCR: 0.05 - 0.15
                # - Spectral centroid: 500 - 3000 Hz
                # - Energy: Above baseline
                # - Rolloff: 2000 - 6000 Hz
                
                speech_score = 0.0
                
                if 0.03 < mean_zcr < 0.20:
                    speech_score += 0.25
                
                if 400 < mean_centroid < 3500:
                    speech_score += 0.25
                
                if 1500 < mean_rolloff < 7000:
                    speech_score += 0.25
                
                if energy > self.noise_baseline * 3.0:
                    speech_score += 0.25
                
                confidence = speech_score
                is_speech = speech_score > 0.6
            
            return {
                'is_speech': is_speech,
                'confidence': confidence,
                'energy': energy,
                'zcr': mean_zcr,
                'spectral_centroid': mean_centroid,
                'spectral_rolloff': mean_rolloff
            }
            
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            return {
                'is_speech': False,
                'confidence': 0.0,
                'energy': 0.0,
                'zcr': 0.0,
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0
            }
    
    def estimate_speaker_count(self, audio: np.ndarray) -> int:
        """
        Estimate number of speakers using advanced techniques.
        
        Args:
            audio: Audio signal
            
        Returns:
            Estimated speaker count
        """
        try:
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
            
            # Calculate spectral features
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            
            # Analyze temporal variation
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Calculate diversity metrics
            mfcc_std = np.std(mfccs, axis=1)
            spectral_diversity = np.mean(np.std(spectral_contrast, axis=1))
            chroma_diversity = np.mean(np.std(chroma, axis=1))
            
            # Temporal variation
            temporal_variation = np.mean(np.abs(mfcc_delta)) + np.mean(np.abs(mfcc_delta2))
            
            # Speaker count estimation
            # Single speaker: low diversity, moderate temporal variation
            # Multiple speakers: high diversity, high temporal variation
            
            diversity_score = (
                np.mean(mfcc_std) / 10.0 +
                spectral_diversity / 5.0 +
                chroma_diversity * 2.0 +
                temporal_variation / 20.0
            )
            
            if diversity_score < 1.5:
                return 1
            elif diversity_score < 3.0:
                return 2
            else:
                return min(3, int(diversity_score / 1.5))
            
        except Exception as e:
            logger.error(f"Speaker count estimation failed: {e}")
            return 1
    
    def detect_whisper(self, audio: np.ndarray) -> bool:
        """
        Detect whispering (low energy, specific frequency characteristics).
        
        Args:
            audio: Audio signal
            
        Returns:
            True if whispering detected
        """
        try:
            energy = np.sum(audio ** 2) / len(audio)
            
            # Whisper characteristics:
            # - Low energy (but above noise)
            # - High frequency content
            # - Low spectral centroid
            
            if self.noise_baseline:
                if self.noise_baseline * 1.5 < energy < self.noise_baseline * 5.0:
                    # Check frequency content
                    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
                    mean_centroid = np.mean(spectral_centroid)
                    
                    # Whisper has lower centroid (500-1500 Hz)
                    if 400 < mean_centroid < 1800:
                        return True
            
            return False
            
        except Exception as e:
            return False
    
    def analyze_audio(self, audio: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze audio frame for speech and violations.
        
        Args:
            audio: Audio signal
            timestamp: Current timestamp
            
        Returns:
            Audio analysis results with violations
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        
        # Establish noise baseline
        self.establish_noise_baseline(audio)
        
        # Filter noise
        filtered_audio = self.filter_noise(audio)
        
        # Detect speech
        speech_result = self.detect_speech_advanced(filtered_audio)
        
        # Estimate speaker count
        speaker_count = 1
        if speech_result['is_speech']:
            speaker_count = self.estimate_speaker_count(filtered_audio)
        
        # Detect whisper
        is_whisper = self.detect_whisper(filtered_audio)
        
        # Add to history
        self.speech_history.append({
            'is_speech': speech_result['is_speech'],
            'speaker_count': speaker_count,
            'confidence': speech_result['confidence'],
            'energy': speech_result['energy'],
            'is_whisper': is_whisper,
            'timestamp': timestamp
        })
        
        # Check for violations
        violations = []
        
        if len(self.speech_history) >= 10:  # At least 1 second
            recent = list(self.speech_history)[-40:]  # Last 4 seconds
            speech_chunks = [s for s in recent if s['is_speech']]
            
            if len(speech_chunks) > 25:  # >60% of time
                if self.speech_start is None:
                    self.speech_start = timestamp
                else:
                    duration = timestamp - self.speech_start
                    if duration >= self.speech_threshold:
                        if timestamp - self.last_violation > self.violation_cooldown:
                            self.last_violation = timestamp
                            
                            # Calculate average speaker count
                            avg_speakers = np.mean([s['speaker_count'] for s in speech_chunks])
                            
                            # Check for whisper
                            whisper_count = sum(1 for s in speech_chunks if s.get('is_whisper', False))
                            is_whispering = whisper_count > len(speech_chunks) * 0.5
                            
                            if avg_speakers > 1.5:
                                violations.append({
                                    'type': 'multiple_speakers',
                                    'speaker_count': int(avg_speakers),
                                    'duration': duration,
                                    'severity': 'high',
                                    'confidence': 0.88,
                                    'message': f'Multiple speakers detected for {duration:.1f} seconds'
                                })
                            elif is_whispering:
                                violations.append({
                                    'type': 'whispering',
                                    'duration': duration,
                                    'severity': 'medium',
                                    'confidence': 0.80,
                                    'message': f'Whispering detected for {duration:.1f} seconds'
                                })
                            else:
                                violations.append({
                                    'type': 'speech_detected',
                                    'duration': duration,
                                    'severity': 'medium',
                                    'confidence': 0.85,
                                    'message': f'Speech detected for {duration:.1f} seconds'
                                })
            else:
                self.speech_start = None
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'speech_detected': speech_result['is_speech'],
            'speaker_count': speaker_count,
            'is_whisper': is_whisper,
            'confidence': speech_result['confidence'],
            'energy': speech_result['energy'],
            'noise_baseline': self.noise_baseline,
            'violations': violations,
            'processing_time': processing_time
        }
    
    def reset(self):
        """Reset all tracking state"""
        self.speech_history.clear()
        self.noise_baseline = None
        self.noise_samples = []
        self.speech_start = None
        self.last_violation = 0
