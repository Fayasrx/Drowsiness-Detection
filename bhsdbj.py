import cv2

# Load Haar cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Constants for smile detection
SMILE_FRAMES_THRESHOLD = 10  # Number of frames where smile should be detected to confirm

# Initialize    
frame_counter = 0

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for the face
        face_ROI_gray = gray[y:y+h, x:x+w]
        face_ROI_color = frame[y:y+h, x:x+w]

        # Detect smiles within the face ROI
        smiles = smile_cascade.detectMultiScale(face_ROI_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        if len(smiles) > 0:  # If a smile is detected
            frame_counter += 1
            for (sx, sy, sw, sh) in smiles:
                # Draw rectangle around the smile
                cv2.rectangle(face_ROI_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
        else:
            # If no smile is detected, reset the frame counter
            frame_counter = 0

        # Check if a smile has been detected for a sufficient number of frames
        if frame_counter >= SMILE_FRAMES_THRESHOLD:
            cv2.putText(frame, "SMILE DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Smile Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
