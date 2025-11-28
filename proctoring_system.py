"""
AI Proctoring System - Detects suspicious behavior during exams
Features:
- Face detection
- Eye gaze tracking
- Looking away detection
- Multiple face detection
- Warning system with logs
"""
import cv2
import numpy as np
from datetime import datetime
import time

class ProctoringSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Tracking variables
        self.looking_away_count = 0
        self.no_face_count = 0
        self.multiple_faces_count = 0
        self.total_warnings = 0
        
        # Thresholds
        self.looking_away_threshold = 30  # frames
        self.no_face_threshold = 20
        self.warning_cooldown = 3  # seconds
        self.last_warning_time = 0
        
        # Log file
        self.log_file = open('proctoring_log.txt', 'a')
        self.log_event("Proctoring session started")
    
    def log_event(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_file.write(log_message)
        self.log_file.flush()
        print(log_message.strip())
    
    def add_warning(self, warning_type):
        current_time = time.time()
        if current_time - self.last_warning_time > self.warning_cooldown:
            self.total_warnings += 1
            self.log_event(f"WARNING #{self.total_warnings}: {warning_type}")
            self.last_warning_time = current_time
    
    def detect_gaze_direction(self, eye_gray, ew):
        """Detect if looking at screen or away"""
        _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                eye_center = ew // 2
                
                # Check if pupil is too far from center
                deviation = abs(cx - eye_center)
                if deviation > ew * 0.3:  # 30% deviation
                    return "AWAY", cx
                else:
                    return "CENTER", cx
        return "UNKNOWN", None
    
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status = "NORMAL"
        status_color = (0, 255, 0)  # Green
        
        # Check for multiple faces
        if len(faces) > 1:
            self.multiple_faces_count += 1
            if self.multiple_faces_count > 10:
                status = "ALERT: Multiple faces detected!"
                status_color = (0, 0, 255)  # Red
                self.add_warning("Multiple people detected in frame")
                self.multiple_faces_count = 0
        else:
            self.multiple_faces_count = 0
        
        # Check for no face
        if len(faces) == 0:
            self.no_face_count += 1
            if self.no_face_count > self.no_face_threshold:
                status = "ALERT: No face detected!"
                status_color = (0, 0, 255)
                self.add_warning("Student left the frame")
                self.no_face_count = 0
        else:
            self.no_face_count = 0
        
        # Process face and eyes
        looking_away = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                away_count = 0
                for (ex, ey, ew, eh) in eyes[:2]:  # Check first 2 eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    direction, cx = self.detect_gaze_direction(eye_gray, ew)
                    
                    if direction == "AWAY":
                        away_count += 1
                        if cx is not None:
                            cv2.circle(roi_color, (ex + cx, ey + eh//2), 3, (0, 0, 255), -1)
                    elif direction == "CENTER" and cx is not None:
                        cv2.circle(roi_color, (ex + cx, ey + eh//2), 3, (0, 255, 0), -1)
                
                # If both eyes looking away
                if away_count >= 2:
                    looking_away = True
                    self.looking_away_count += 1
                    if self.looking_away_count > self.looking_away_threshold:
                        status = "ALERT: Looking away from screen!"
                        status_color = (0, 165, 255)  # Orange
                        self.add_warning("Student looking away from screen")
                        self.looking_away_count = 0
                else:
                    self.looking_away_count = max(0, self.looking_away_count - 2)
        
        return frame, status, status_color
    
    def run(self):
        webcam = cv2.VideoCapture(0)
        print("\n" + "="*60)
        print("AI PROCTORING SYSTEM ACTIVE")
        print("="*60)
        print("Press ESC to end session")
        print("="*60 + "\n")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            
            frame, status, status_color = self.process_frame(frame)
            
            # Display info panel
            cv2.rectangle(frame, (10, 10), (630, 120), (0, 0, 0), -1)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"Total Warnings: {self.total_warnings}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Warning indicator
            if self.total_warnings > 0:
                warning_text = f"WARNINGS: {self.total_warnings}"
                cv2.putText(frame, warning_text, (frame.shape[1] - 200, 40), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("AI Proctoring System", frame)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
        
        self.log_event(f"Proctoring session ended. Total warnings: {self.total_warnings}")
        webcam.release()
        cv2.destroyAllWindows()
        self.log_file.close()

if __name__ == "__main__":
    proctor = ProctoringSystem()
    proctor.run()
