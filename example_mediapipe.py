"""
Alternative gaze tracking using MediaPipe (easier to install on Windows)
"""
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

webcam = cv2.VideoCapture(0)

# Eye landmark indices for MediaPipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_eye_position(iris_center, eye_center):
    """Determine if looking left, right, or center"""
    if iris_center is None or eye_center is None:
        return "Unknown"
    
    diff = iris_center[0] - eye_center[0]
    if diff < -0.015:
        return "Looking right"
    elif diff > 0.015:
        return "Looking left"
    else:
        return "Looking center"

print("Starting webcam... Press ESC to exit")

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    text = "No face detected"
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get left iris center
        left_iris_x = np.mean([landmarks[i].x for i in LEFT_IRIS])
        left_iris_y = np.mean([landmarks[i].y for i in LEFT_IRIS])
        
        # Get left eye center
        left_eye_x = np.mean([landmarks[i].x for i in LEFT_EYE])
        left_eye_y = np.mean([landmarks[i].y for i in LEFT_EYE])
        
        # Draw iris
        iris_x = int(left_iris_x * w)
        iris_y = int(left_iris_y * h)
        cv2.circle(frame, (iris_x, iris_y), 3, (0, 255, 0), -1)
        
        # Determine gaze direction
        text = get_eye_position((left_iris_x, left_iris_y), (left_eye_x, left_eye_y))
    
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Gaze Tracking", frame)
    
    if cv2.waitKey(1) == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
