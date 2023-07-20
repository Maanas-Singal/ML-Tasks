import cv2

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and draw rectangles around faces
def detect_faces(image_path):
    # Read the image from file
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale (Haar Cascade works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image using the Haar Cascade model
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the image with rectangles around faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with the path to your image
    detect_faces(image_path)

### Make sure to replace "path/to/your/image.jpg" with the actual file path of the image you want to use for face detection.
