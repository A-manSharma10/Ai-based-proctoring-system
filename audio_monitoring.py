"""
AUDIO MONITORING MODULE FOR PROCTORING SYSTEM
Detects multiple voices, background noise, and suspicious audio patterns
"""

import pyaudio
import numpy as np
import wave
import os
import datetime
import time
from collections import deque
import threading

class AudioMonitor:
    def __init__(self):
        print("\n" + "="*80)
        print("AUDIO MONITORING SYSTEM")
        print("="*80)
        print("\nInitializing audio detection...")
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 1
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Thresholds
        self.SILENCE_THRESHOLD = 500  # RMS threshold for silence
        self.SPEECH_THRESHOLD = 1500  # RMS threshold for speech
        self.MULTIPLE_VOICE_THRESHOLD = 3000  # RMS threshold for multiple voices
        self.NOISE_THRESHOLD = 2500  # Background noise threshold
        
        # Counters
        self.speech_counter = 0
        self.multiple_voice_counter = 0
        self.noise_counter = 0
        self.silence_counter = 0
        
        # Thresholds for violations
        self.SPEECH_VIOLATION_THRESHOLD = 30  # 30 chunks (~30 seconds)
        self.MULTIPLE_VOICE_VIOLATION_THRESHOLD = 15  # 15 chunks (~15 seconds)
        self.NOISE_VIOLATION_THRESHOLD = 20  # 20 chunks (~20 seconds)
        
        # Buffers
        self.audio_buffer = deque(maxlen=50)
        self.rms_buffer = deque(maxlen=10)
        
        # Violation tracking
        self.violations = []
        self.violation_counts = {
            'SPEECH_DETECTED': 0,
            'MULTIPLE_VOICES': 0,
            'BACKGROUND_NOISE': 0,
            'SUSPICIOUS_AUDIO': 0
        }
        
        # Recording
        self.recording = False
        self.frames = []
        os.makedirs('audio_violations', exist_ok=True)
        
        # Session
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Status
        self.monitoring = False
        self.last_violation_time = 0
        self.violation_cooldown = 5.0  # 5 seconds
        
        print("Audio monitoring initialized successfully")
        print("="*80 + "\n")
    
    def start_monitoring(self):
        """Start audio monitoring"""
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            self.monitoring = True
            print("Audio monitoring started")
            return True
        except Exception as e:
            print(f"Error starting audio monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.monitoring = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Audio monitoring stopped")
    
    def calculate_rms(self, data):
        """Calculate RMS (Root Mean Square) of audio data"""
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms
    
    def calculate_frequency_spectrum(self, data):
        """Calculate frequency spectrum using FFT"""
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        return magnitude
    
    def detect_multiple_voices(self, rms, spectrum):
        """
        Detect multiple voices using:
        1. High RMS (loud audio)
        2. Multiple frequency peaks (different voice frequencies)
        """
        # Check RMS level
        if rms < self.SPEECH_THRESHOLD:
            return False
        
        # Find peaks in frequency spectrum
        # Human voice typically 85-255 Hz (fundamental) + harmonics
        low_freq = spectrum[10:50]   # ~85-430 Hz
        mid_freq = spectrum[50:150]  # ~430-1290 Hz
        high_freq = spectrum[150:300] # ~1290-2580 Hz
        
        # Count significant peaks
        low_peaks = np.sum(low_freq > np.mean(low_freq) * 2)
        mid_peaks = np.sum(mid_freq > np.mean(mid_freq) * 2)
        high_peaks = np.sum(high_freq > np.mean(high_freq) * 2)
        
        total_peaks = low_peaks + mid_peaks + high_peaks
        
        # Multiple voices have more frequency peaks and higher RMS
        if total_peaks > 15 and rms > self.MULTIPLE_VOICE_THRESHOLD:
            return True
        
        return False
    
    def detect_speech(self, rms):
        """Detect if speech is present"""
        return rms > self.SPEECH_THRESHOLD
    
    def detect_background_noise(self, rms, spectrum):
        """Detect background noise (TV, music, etc.)"""
        # Background noise has consistent high RMS across frequencies
        if rms > self.NOISE_THRESHOLD:
            # Check if energy is spread across frequencies (noise characteristic)
            low_energy = np.mean(spectrum[10:50])
            mid_energy = np.mean(spectrum[50:150])
            high_energy = np.mean(spectrum[150:300])
            
            # Noise has more uniform energy distribution
            energy_variance = np.var([low_energy, mid_energy, high_energy])
            
            if energy_variance < np.mean([low_energy, mid_energy, high_energy]) * 0.5:
                return True
        
        return False
    
    def log_violation(self, vtype, severity, details=""):
        """Log audio violation"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_violation_time < self.violation_cooldown:
            return
        
        self.last_violation_time = current_time
        
        print(f"\nAUDIO VIOLATION: {vtype} [{severity}]")
        if details:
            print(f"   {details}")
        
        self.violations.append({
            'type': vtype,
            'severity': severity,
            'timestamp': datetime.datetime.now(),
            'details': details
        })
        self.violation_counts[vtype] += 1
        
        # Start recording violation audio
        if not self.recording:
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio for violation evidence"""
        self.recording = True
        self.frames = []
        print("   Recording audio evidence...")
    
    def stop_recording(self):
        """Stop recording and save audio"""
        if self.recording and len(self.frames) > 0:
            self.recording = False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_violations/session_{self.session_id}_{timestamp}.wav"
            
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            print(f"   Audio saved: {filename}")
            self.frames = []
    
    def process_audio(self):
        """Process audio chunk and detect violations"""
        if not self.monitoring or not self.stream:
            return None
        
        try:
            # Read audio data
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Record if violation is active
            if self.recording:
                self.frames.append(data)
                if len(self.frames) > 50:  # ~50 seconds max
                    self.stop_recording()
            
            # Calculate RMS
            rms = self.calculate_rms(data)
            self.rms_buffer.append(rms)
            
            # Calculate frequency spectrum
            spectrum = self.calculate_frequency_spectrum(data)
            
            # Get average RMS
            avg_rms = np.mean(self.rms_buffer) if len(self.rms_buffer) > 0 else rms
            
            # Detect violations
            status = "SILENT"
            violation = None
            
            # Check for multiple voices
            if self.detect_multiple_voices(rms, spectrum):
                self.multiple_voice_counter += 1
                self.speech_counter = 0
                self.noise_counter = 0
                status = "MULTIPLE VOICES"
                
                if self.multiple_voice_counter >= self.MULTIPLE_VOICE_VIOLATION_THRESHOLD:
                    violation = {
                        'type': 'MULTIPLE_VOICES',
                        'severity': 'CRITICAL',
                        'details': f'Multiple people detected speaking (RMS: {int(rms)})'
                    }
                    self.multiple_voice_counter = 0
            
            # Check for background noise
            elif self.detect_background_noise(rms, spectrum):
                self.noise_counter += 1
                self.speech_counter = 0
                self.multiple_voice_counter = 0
                status = "BACKGROUND NOISE"
                
                if self.noise_counter >= self.NOISE_VIOLATION_THRESHOLD:
                    violation = {
                        'type': 'BACKGROUND_NOISE',
                        'severity': 'MEDIUM',
                        'details': f'Continuous background noise detected (RMS: {int(rms)})'
                    }
                    self.noise_counter = 0
            
            # Check for speech
            elif self.detect_speech(rms):
                self.speech_counter += 1
                self.multiple_voice_counter = 0
                self.noise_counter = 0
                status = "SPEECH"
                
                if self.speech_counter >= self.SPEECH_VIOLATION_THRESHOLD:
                    violation = {
                        'type': 'SPEECH_DETECTED',
                        'severity': 'MEDIUM',
                        'details': f'Prolonged speech detected (RMS: {int(rms)})'
                    }
                    self.speech_counter = 0
            
            # Silence
            else:
                self.silence_counter += 1
                self.speech_counter = max(0, self.speech_counter - 1)
                self.multiple_voice_counter = max(0, self.multiple_voice_counter - 1)
                self.noise_counter = max(0, self.noise_counter - 1)
                status = "SILENT"
            
            # Log violation if detected
            if violation:
                self.log_violation(violation['type'], violation['severity'], violation['details'])
            
            return {
                'status': status,
                'rms': int(rms),
                'avg_rms': int(avg_rms),
                'speech_level': self.speech_counter,
                'multiple_voice_level': self.multiple_voice_counter,
                'noise_level': self.noise_counter
            }
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    def generate_report(self):
        """Generate audio monitoring report"""
        report_path = f"audio_violations/audio_report_{self.session_id}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AUDIO MONITORING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            end_time = datetime.datetime.now()
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {str(end_time - self.session_start).split('.')[0]}\n\n")
            
            f.write(f"Total Audio Violations: {len(self.violations)}\n\n")
            
            if len(self.violations) > 0:
                f.write("Violation Summary:\n")
                for vtype, count in sorted(self.violation_counts.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        f.write(f"  - {vtype}: {count}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("DETAILED VIOLATION LOG\n")
                f.write("="*80 + "\n\n")
                
                for i, v in enumerate(self.violations, 1):
                    f.write(f"{i}. {v['type']} [{v['severity']}]\n")
                    f.write(f"   Time: {v['timestamp'].strftime('%H:%M:%S')}\n")
                    if v['details']:
                        f.write(f"   Details: {v['details']}\n")
                    f.write("\n")
        
        print(f"\nAudio report saved: {report_path}")
        return report_path


def test_audio_monitoring():
    """Test audio monitoring standalone"""
    monitor = AudioMonitor()
    
    if not monitor.start_monitoring():
        print("Failed to start audio monitoring")
        return
    
    print("\nMonitoring audio... Press Ctrl+C to stop\n")
    print("Status will update every second:")
    print("-" * 80)
    
    try:
        while True:
            result = monitor.process_audio()
            
            if result:
                # Display status
                print(f"\rStatus: {result['status']:20} | "
                      f"RMS: {result['rms']:5} | "
                      f"Avg: {result['avg_rms']:5} | "
                      f"Speech: {result['speech_level']:3} | "
                      f"Multi: {result['multiple_voice_level']:3} | "
                      f"Noise: {result['noise_level']:3}", end='')
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Stopping audio monitoring...")
        monitor.stop_recording()
        monitor.stop_monitoring()
        monitor.generate_report()
        
        print("\n" + "="*80)
        print("AUDIO MONITORING SESSION SUMMARY")
        print("="*80)
        print(f"Total Violations: {len(monitor.violations)}")
        
        if len(monitor.violations) > 0:
            print("\nViolation Breakdown:")
            for vtype, count in sorted(monitor.violation_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  - {vtype}: {count}")
        
        print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUDIO MONITORING MODULE - STANDALONE TEST")
    print("="*80)
    print("\nThis module detects:")
    print("  - Speech/talking")
    print("  - Multiple voices (conversations)")
    print("  - Background noise (TV, music, etc.)")
    print("\nViolations are logged and audio evidence is recorded.")
    print("="*80 + "\n")
    
    test_audio_monitoring()
