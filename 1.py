import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import keyboard

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # Pre-trained model (nano version)

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Dictionary to map class index to label name
CLASS_NAMES = model.names

# Function to display image using Matplotlib
def display_image(frame):
    # Resize the frame to be smaller
    scale_percent = 30  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Create a blank image for placing the resized frame in the right corner
    blank_image = np.zeros((720, 1280, 3), np.uint8)
    x_offset = blank_image.shape[1] - resized_frame.shape[1] - 110  # Adjusted offset to move left
    y_offset = 10

    blank_image[y_offset:y_offset+resized_frame.shape[0], x_offset:x_offset+resized_frame.shape[1]] = resized_frame

    plt.imshow(cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

# Business logic to determine the alert based on detections
def business_logic(detections, faces):
    cell_phone_count = 0
    face_detected = len(faces) > 0

    for detection in detections:
        bbox, score, class_id = detection[:4], detection[4], int(detection[5])
        label = CLASS_NAMES[class_id]

        if label == "cell phone" and score > 0.5:
            cell_phone_count += 1

    if not face_detected:
        return "no_face_detected"
    elif cell_phone_count >= 3:
        return "exam_terminated"
    elif cell_phone_count > 0:
        return "caution"
    else:
        return "safe"

# Function to display an alert on the screen
def trigger_alert(alert_type):
    if alert_type == "exam_terminated":
        message = "ALERT: Malpractice detected multiple times!\nYour exam has been terminated."
        color = 'red'
    elif alert_type == "caution":
        message = "CAUTION: Cell Phone Detected!"
        color = 'yellow'
    elif alert_type == "no_face_detected":
        message = "ALERT: No Face Detected!"
        color = 'blue'
    else:
        return  # No alert needed

    # Display the alert in full-screen mode
    plt.figure(figsize=(8, 6))
    fig_manager = plt.get_current_fig_manager()
    fig_manager.full_screen_toggle()
    plt.text(0.5, 0.5, message, fontsize=20, ha='center', va='center', color=color)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(3)  # Display for 3 seconds
    plt.close()

# Main loop to perform object detection and trigger alerts
cell_phone_detection_count = 0
full_screen_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not full_screen_mode:
        # Make the display full-screen at the start
        plt.figure(figsize=(8, 6))
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        full_screen_mode = True

    # Perform object detection
    results = model(frame)

    # Extract detections from results
    detections = results[0].boxes.data.cpu().numpy() if hasattr(results[0], 'boxes') else np.array([])

    # Perform face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply business logic based on detections and faces
    alert_type = business_logic(detections, faces)

    # Update cell phone detection count
    if alert_type == "caution":
        cell_phone_detection_count += 1
        if cell_phone_detection_count >= 3:
            alert_type = "exam_terminated"

    # Trigger appropriate alert
    trigger_alert(alert_type)

    if alert_type == "exam_terminated":
        break

    # Display the frame using Matplotlib
    display_image(frame)

    # Break loop on 'q' key press
    if keyboard.is_pressed('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
