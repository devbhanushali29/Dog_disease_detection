import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import numpy as np
import uuid

app = Flask(__name__)
app.secret_key = "dog_skin_disease_detection"

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLOv8 model
model = YOLO('models/yolov8_trained_model_2.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid overwriting
        unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)
        
        # Process the image with YOLOv8
        results = model(upload_path)
        result = results[0]  # Get the first result
        
        # Save the result image with bounding boxes
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Save the result image with bounding boxes
        result_img = result.plot()
        Image.fromarray(result_img).save(result_path)
        
        # Get detection details
        boxes = result.boxes
        class_names = model.names
        
        detections = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = class_names[class_id]
            confidence = round(box.conf[0].item() * 100, 2)
            detections.append({
                'class_name': class_name,
                'confidence': confidence
            })
        
        return render_template('result.html', 
                              result_image=result_filename, 
                              original_image=unique_filename,
                              detections=detections)
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
