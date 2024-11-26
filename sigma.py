import time
from pickle import FALSE
import cv2
import webbrowser

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Flag to track if the URL has already been opened
url_opened = False

while True:
    # Read frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=18, minSize=(10, 10))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Check if more than one face is detected
    num_faces = len(faces)

    if num_faces > 1 and not url_opened:
        print(f"{num_faces} faces detected")
        # Open Google Classroom link in Chrome
        webbrowser.open('https://classroom.google.com', new=2)  # new=2 opens in a new tab if possible
        url_opened = True  # Prevent opening the URL multiple times

    # If no faces are detected, reset url_opened to allow for the next detection
    elif num_faces == 0:
        url_opened = False

    # Display the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()