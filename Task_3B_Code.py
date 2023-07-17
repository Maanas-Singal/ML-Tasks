# Importing CV2 Python Library
import cv2

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the video file path

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a mask with the same size as the frame
    mask = cv2.GaussianBlur(frame, (0, 0), sigmaX=30)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        # Clear the corresponding region in the mask
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

    # Apply the mask to the frame to blur or highlight the non-face regions
    result = cv2.bitwise_and(frame, mask)

    # Display the resulting frame
    cv2.imshow('Face Detection', result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
