# Importing the required Python Libraries
import cv2
import numpy as np

# Define the physical dimensions of the face (in meters)
face_width = 0.15  # Example: assuming the average face width is 15cm

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the video file path

# Get the focal length of the camera (assumed to be known or calibrated)
focal_length = None  # Set the focal length here (in pixels)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Calculate the distance using the formula: distance = (face_width * focal_length) / face_width_in_pixels
        distance = (face_width * focal_length) / w

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
