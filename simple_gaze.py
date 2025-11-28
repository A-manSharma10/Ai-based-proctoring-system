"""
Simple gaze tracking using only OpenCV (no dlib needed)
"""
import cv2
import numpy as np

# Load face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)

print("Starting webcam... Press ESC to exit")

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    text = "No face detected"
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            text = "Eyes detected"
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Get eye region
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Find pupil (darkest point)
                _, threshold = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour (pupil)
                    contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(roi_color, (ex + cx, ey + cy), 3, (0, 0, 255), -1)
                        
                        # Determine gaze direction
                        eye_center = ew // 2
                        if cx < eye_center - 5:
                            text = "Looking LEFT"
                        elif cx > eye_center + 5:
                            text = "Looking RIGHT"
                        else:
                            text = "Looking CENTER"
    
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Simple Gaze Tracking", frame)
    
    if cv2.waitKey(1) == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
