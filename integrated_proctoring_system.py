"""
INTEGRATED PROCTORING SYSTEM
Combines video monitoring + audio monitoring for complete exam proctoring
"""

import cv2
import threading
import time
from enhanced_professional_proctoring import EnhancedProfessionalProctoring
from audio_monitoring import AudioMonitor

class IntegratedProctoringSystem:
    def __init__(self):
        print("\n" + "="*80)
        print("INTEGRATED PROCTORING SYSTEM")
        print("Video + Audio Monitoring")
        print("="*80 + "\n")
        
        # Initialize video monitoring
        print("Initializing video monitoring...")
        self.video_monitor = EnhancedProfessionalProctoring()
        
        # Initialize audio monitoring
        print("\nInitializing audio monitoring...")
        self.audio_monitor = AudioMonitor()
        
        # Audio thread
        self.audio_thread = None
        self.audio_running = False
        
        # Combined violations
        self.total_violations = 0
        
        print("\n" + "="*80)
        print("INTEGRATED SYSTEM READY")
        print("="*80 + "\n")
    
    def audio_monitoring_thread(self):
        """Run audio monitoring in separate thread"""
        while self.audio_running:
            audio_result = self.audio_monitor.process_audio()
            
            if audio_result and audio_result['status'] != 'SILENT':
                # Update display with audio status
                self.current_audio_status = audio_result
            
            time.sleep(0.1)
    
    def start_audio_monitoring(self):
        """Start audio monitoring in background thread"""
        if self.audio_monitor.start_monitoring():
            self.audio_running = True
            self.audio_thread = threading.Thread(target=self.audio_monitoring_thread, daemon=True)
            self.audio_thread.start()
            print("Audio monitoring started in background")
            return True
        return False
    
    def stop_audio_monitoring(self):
        """Stop audio monitoring"""
        self.audio_running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        self.audio_monitor.stop_recording()
        self.audio_monitor.stop_monitoring()
    
    def draw_audio_panel(self, frame, audio_status):
        """Draw audio monitoring panel on video frame"""
        h, w = frame.shape[:2]
        
        # Audio panel position (top right, below detection panel)
        panel_x = w - 330
        panel_y = 370
        panel_w = 320
        panel_h = 180
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (40, 40, 50), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (0, 255, 150), 2)
        
        # Title
        cv2.putText(frame, "AUDIO MONITOR", (panel_x + 15, panel_y + 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 150), 2)
        cv2.line(frame, (panel_x + 15, panel_y + 45), (panel_x + panel_w - 15, panel_y + 45), 
                (0, 255, 150), 2)
        
        y_pos = panel_y + 75
        
        if audio_status:
            # Status
            status = audio_status['status']
            status_color = (0, 255, 150)  # Green
            
            if status == "SPEECH":
                status_color = (0, 165, 255)  # Orange
            elif status == "MULTIPLE VOICES":
                status_color = (0, 0, 255)  # Red
            elif status == "BACKGROUND NOISE":
                status_color = (0, 165, 255)  # Orange
            
            cv2.putText(frame, f"Status: {status}", (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            y_pos += 30
            
            # RMS Level
            rms = audio_status['rms']
            cv2.putText(frame, f"Level: {rms}", (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_pos += 30
            
            # Counters
            if audio_status['multiple_voice_level'] > 0:
                cv2.putText(frame, f"Multi-Voice: {audio_status['multiple_voice_level']}/15", 
                           (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_pos += 25
            
            if audio_status['speech_level'] > 0:
                cv2.putText(frame, f"Speech: {audio_status['speech_level']}/30", 
                           (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                y_pos += 25
            
            if audio_status['noise_level'] > 0:
                cv2.putText(frame, f"Noise: {audio_status['noise_level']}/20", 
                           (panel_x + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            cv2.putText(frame, "Status: MONITORING", (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 150), 1)
        
        return frame
    
    def run(self):
        """Run integrated monitoring system"""
        print("Starting integrated proctoring system...")
        print("Calibrating video... Look at screen for 3 seconds\n")
        
        # Start audio monitoring
        self.start_audio_monitoring()
        
        calibration_frames = 0
        calibration_required = 90
        self.current_audio_status = None
        
        try:
            while True:
                ret, frame = self.video_monitor.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Calculate FPS
                fps = self.video_monitor.calculate_fps()
                
                # Detect faces
                faces = self.video_monitor.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                )
                
                # Calibration phase
                if not self.video_monitor.calibrated and len(faces) > 0:
                    calibration_frames += 1
                    if calibration_frames >= calibration_required:
                        self.video_monitor.calibrate(faces[0])
                
                # Monitoring phase
                if self.video_monitor.calibrated:
                    # Video monitoring logic (simplified from main system)
                    if len(faces) == 0:
                        self.video_monitor.no_face_counter += 1
                        if self.video_monitor.no_face_counter >= self.video_monitor.NO_FACE_THRESHOLD:
                            self.video_monitor.log_violation('NO_FACE', 'CRITICAL', 'Face not detected')
                            self.video_monitor.no_face_counter = 0
                    elif len(faces) > 1:
                        self.video_monitor.multiple_face_counter += 1
                        if self.video_monitor.multiple_face_counter >= self.video_monitor.MULTIPLE_FACE_THRESHOLD:
                            self.video_monitor.log_violation('MULTIPLE_FACES', 'CRITICAL', f'{len(faces)} faces detected')
                            self.video_monitor.multiple_face_counter = 0
                    else:
                        # Single face detected
                        self.video_monitor.no_face_counter = 0
                        self.video_monitor.multiple_face_counter = 0
                        
                        face = faces[0]
                        x, y, w, h = face
                        
                        # Draw face rectangle
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 2)
                        
                        # Eye and pupil detection
                        eyes = self.video_monitor.detect_eyes_and_pupils(frame, face)
                        
                        # Gaze tracking
                        if len(eyes) >= 2:
                            looking_away, eyes_closed = self.video_monitor.calculate_gaze_direction(eyes, w, h)
                            
                            if eyes_closed:
                                self.video_monitor.eyes_closed_counter += 1
                                if self.video_monitor.eyes_closed_counter >= self.video_monitor.EYES_CLOSED_THRESHOLD:
                                    self.video_monitor.log_violation('EYES_CLOSED', 'MEDIUM', 'Eyes closed for extended period')
                                    self.video_monitor.eyes_closed_counter = 0
                            else:
                                self.video_monitor.eyes_closed_counter = 0
                            
                            if looking_away:
                                self.video_monitor.looking_away_counter += 1
                                if self.video_monitor.looking_away_counter >= self.video_monitor.LOOKING_AWAY_THRESHOLD:
                                    self.video_monitor.log_violation('LOOKING_AWAY', 'MEDIUM', f'Gaze: {self.video_monitor.gaze_direction}')
                                    self.video_monitor.looking_away_counter = 0
                            else:
                                self.video_monitor.looking_away_counter = 0
                        
                        # Distance check
                        distance_violation = self.video_monitor.check_distance(w * h)
                        if distance_violation:
                            self.video_monitor.log_violation(distance_violation, 'MEDIUM', 'Distance from camera')
                        
                        # Profile detection
                        profiles = self.video_monitor.profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                        if len(profiles) > 0:
                            self.video_monitor.log_violation('PROFILE_DETECTED', 'MEDIUM', 'Side face detected')
                    
                    # Object detection (every 20 frames)
                    if self.video_monitor.frame_count % self.video_monitor.OBJECT_DETECTION_INTERVAL == 0:
                        detected_objects = self.video_monitor.detect_objects_yolo(frame)
                        for obj in detected_objects:
                            self.video_monitor.log_violation(f'{obj}_DETECTED', 'CRITICAL', f'{obj} detected in frame')
                
                # Draw UI
                status = "Calibrating..." if not self.video_monitor.calibrated else "Monitoring Active"
                frame = self.video_monitor.draw_ui(frame, status, [], fps)
                
                # Draw audio panel
                frame = self.draw_audio_panel(frame, self.current_audio_status)
                
                # Draw popups
                frame = self.video_monitor.draw_popups(frame)
                
                # Check for max violations
                if self.video_monitor.critical_violation_count >= self.video_monitor.max_critical_violations:
                    # Show termination screen
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, "EXAM TERMINATED", (frame.shape[1]//2 - 250, frame.shape[0]//2 - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, "Maximum violations reached", (frame.shape[1]//2 - 200, frame.shape[0]//2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imshow('Integrated Proctoring System', frame)
                    cv2.waitKey(3000)
                    break
                
                # Display
                cv2.imshow('Integrated Proctoring System', frame)
                
                self.video_monitor.frame_count += 1
                
                # Exit on ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
        
        finally:
            # Cleanup
            print("\n" + "="*80)
            print("SESSION ENDED")
            print("="*80 + "\n")
            
            # Stop audio monitoring
            self.stop_audio_monitoring()
            
            # Generate reports
            video_report = self.video_monitor.generate_report()
            audio_report = self.audio_monitor.generate_report()
            
            # Combined summary
            total_violations = len(self.video_monitor.violations) + len(self.audio_monitor.violations)
            
            print(f"\nVideo violations: {len(self.video_monitor.violations)}")
            print(f"Audio violations: {len(self.audio_monitor.violations)}")
            print(f"Total violations: {total_violations}")
            
            print("\n" + "="*80)
            
            # Cleanup
            self.video_monitor.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        system = IntegratedProctoringSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
