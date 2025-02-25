from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_cors import CORS


# Check PyTorch version
import torch
print("PyTorch version:", torch.__version__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load YOLOv5 Model
model_path = r"C:\Users\KINJAL\OneDrive\Desktop\Human_Face_Detection\runs\content\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)  # Load YOLO model

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to process image and return detections
def detect_faces(image_path):
    results = model.predict(image_path)  # Run YOLO model on image path

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence,
                "class": class_id
            })
    
    # Draw bounding boxes
    img = cv2.imread(image_path)
    for det in detections:
        cv2.rectangle(img, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
    cv2.imwrite(output_path, img)

    return detections, output_path

# API Endpoint for Face Detection
@app.route('/detect', methods=['POST'])
def upload_file():
    print(request.files)
    if '' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    detections, output_path = detect_faces(filepath)

    return jsonify({
        "detections": detections,
        "output_image": output_path
    })

# Run Flask Server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000) 
