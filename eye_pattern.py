import cv2
import dlib

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables for blink detection
is_blinking = False
blink_counter = 0
ear_threshold = 0.2  # Adjust this threshold as needed

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/pranavdhawan/Projects/blinkit/shape_predictor_68_face_landmarks.dat')  # Provide the path to the shape predictor file

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    a = distance(eye[1], eye[5])
    b = distance(eye[2], eye[4])
    c = distance(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return ((p1 - p2) ** 2) ** 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get the landmarks for the detected face
        landmarks = predictor(gray, face)

        # Extract the eye landmarks
        left_eye_landmarks = [landmarks.part(i) for i in range(36, 42)]
        right_eye_landmarks = [landmarks.part(i) for i in range(42, 48)]

        # Calculate the EAR for each eye
        left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
        right_eye_ear = eye_aspect_ratio(right_eye_landmarks)

        # Calculate the average EAR for both eyes
        average_ear = (left_eye_ear + right_eye_ear) / 2.0

        # Display the EAR on the frame
        cv2.putText(frame, f'EAR: {average_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # If the average EAR falls below the threshold, consider it a blink
        if average_ear < ear_threshold:
            is_blinking = True
        else:
            if is_blinking:
                blink_counter += 1
                is_blinking = False

    # Display the blink count on the frame
    cv2.putText(frame, f'Blink Count: {blink_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
