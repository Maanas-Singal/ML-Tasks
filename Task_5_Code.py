# Import The CV2 Python Library
import cv2

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the accessory image (e.g., sunglasses)
accessory_image = cv2.imread('sunglasses.png', -1)  # Replace 'sunglasses.png' with the path to your accessory image

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the video file path

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Resize the accessory image to fit the face
        resized_accessory = cv2.resize(accessory_image, (w, h))

        # Create an alpha mask from the accessory image
        alpha = resized_accessory[:, :, 3] / 255.0

        # Add the accessory to the frame
        for c in range(0, 3):
            frame[y:y + h, x:x + w, c] = (1 - alpha) * frame[y:y + h, x:x + w, c] + alpha * resized_accessory[:, :, c]

    # Display the resulting frame
    cv2.imshow('Face Accessories', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
