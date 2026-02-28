import cv2
import numpy as np
from scipy.spatial import distance

# Function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Threshold for eye aspect ratio below which the person is considered drowsy
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20  # Number of consecutive frames a person needs to have drowsy eyes to trigger an alarm

cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # ROI (Region of Interest) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        ear_total = 0
        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey+eh, ex:ex+ew]
            # Draw rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Assume the detected eyes are split evenly between left and right eyes for simplicity
            if len(eyes) == 2:
                # Approximate the coordinates of the eye landmarks
                left_eye = np.array([[ex, ey], [ex, ey+eh//2], [ex+ew//2, ey+eh//2], 
                                     [ex+ew//2, ey], [ex+ew, ey], [ex+ew, ey+eh//2]])
                
                right_eye = np.array([[ex, ey], [ex, ey+eh//2], [ex+ew//2, ey+eh//2], 
                                      [ex+ew//2, ey], [ex+ew, ey], [ex+ew, ey+eh//2]])
                
                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear_total = (left_ear + right_ear) / 2.0
        
        if ear_total < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            frame_counter = 0

    # Show the frame with face and eye detection
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()