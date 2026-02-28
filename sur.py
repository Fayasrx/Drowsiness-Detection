import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# Initialize variables
COUNTER = 0

# Load the facial landmark predictor and detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib.shape_predictor('shape_predictor_68_face_landmarks.dat'))

# Start video stream
vs = VideoStream(src=0).start()

# Loop over frames from the video stream
while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray, 0)
    
    for rect in rectangles:
        shape = predictor(gray, rect)
        shape = np.array([(p.x, p.y) for p in shape.parts()])
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_eye_ear = eye_aspect_ratio(left_eye)
        right_eye_ear = eye_aspect_ratio(right_eye)
        ear = (left_eye_ear + right_eye_ear) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            COUNTER = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()