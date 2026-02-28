import cv2
import numpy as np
from scipy.spatial import distance as dist

# EAR function to compute the eye aspect ratio
def calculate_EAR(eye):
    # Calculate the vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Calculate the horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Load the face and eye cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define constants for sleep detection
EAR_THRESHOLD = 0.2  # Threshold for the EAR value
FRAMES_THRESHOLD = 15  # Number of frames for which EAR should be below the threshold

# Initialize variables for counting frames
frame_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest (ROI) for face
        roi_color = frame[y:y + h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes within the face ROI
        
        for (ex, ey, ew, eh) in eyes[:2]:  # Consider the first two eyes detected
            eye = roi_color[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Optional: Compute EAR using landmarks (if landmarks are available)
            # For simplicity, we're focusing on eye detection only

        # Placeholder for EAR calculation, assuming we have landmarks
        # ear = calculate_EAR(eye_points)
        # For now, we will use a dummy EAR value
        ear = 0.15  # Simulating closed eyes
        
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= FRAMES_THRESHOLD:
                cv2.putText(frame, "SLEEP DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame_counter = 0
    
    # Display the resulting frame
    cv2.imshow('Sleep Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()