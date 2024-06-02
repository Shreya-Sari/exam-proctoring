from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # Pre-trained model (nano version)
CLASS_NAMES = model.names

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Endpoint to receive image data and perform object detection
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    nparr = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(frame)

    # Extract detections from results
    detections = results[0].boxes.data.cpu().numpy() if hasattr(results[0], 'boxes') else np.array([])

    # Perform face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply business logic based on detections and faces
    alert_type = business_logic(detections, faces)

    return jsonify({'alert': alert_type})

if __name__ == '__main__':
    app.run(debug=True)
