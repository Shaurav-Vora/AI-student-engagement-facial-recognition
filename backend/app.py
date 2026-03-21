# FILE: backend/app.py
import os
import sys
import cv2
import json
import time
import threading
import traceback
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Ensure the backend directory is in sys.path for imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from pipeline import ClassEngagementPipeline

app = Flask(__name__)
# Enable CORS for the React frontend (running on Vite's default port 5173)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global pipeline instance
pipeline = None
# Global state for latest frame and student data
latest_annotated_frame = None
latest_students_data = []
# Lock for thread-safe access to globals
lock = threading.Lock()

def start_pipeline():
    """Background thread function that runs the webcam loop."""
    global pipeline, latest_annotated_frame, latest_students_data, lock
    
    try:
        os.makedirs(os.path.join(BACKEND_DIR, "database"), exist_ok=True)
        pipeline = ClassEngagementPipeline()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam for backend API.")
            return

        print("Started background video pipeline thread.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame
            context = pipeline.process_frame(frame)
            annotated_frame = pipeline.annotate_frame(context)
            
            # Serialize student data (DTOs to dicts)
            students_data = []
            for s in context.students:
                students_data.append({
                    "name": s.name,
                    "fer_emotion": s.fer_emotion,
                    "engagement_state": s.engagement_state,
                    "confidence": s.confidence,
                    "bounding_box": s.bounding_box
                })

            # Update globals safely
            with lock:
                latest_annotated_frame = annotated_frame.copy()
                latest_students_data = students_data
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"PIPELINE THREAD CRASHED!")
        print(f"{'='*50}")
        traceback.print_exc()
        print(f"{'='*50}\n")

def generate_mjpeg():
    """Generator for the MJPEG video feed."""
    global latest_annotated_frame, lock
    
    while True:
        with lock:
            if latest_annotated_frame is None:
                # Need to release the lock and sleep so we don't block the thread
                frame = None
            else:
                frame = latest_annotated_frame.copy()
                
        if frame is None:
            time.sleep(0.05)
            continue
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the multipart boundary and the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- API ENDPOINTS ---

@app.route('/video_feed')
def video_feed():
    """Returns the live MJPEG video stream."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/students')
def get_students():
    """Returns the latest detection and engagement data as JSON."""
    global latest_students_data, lock
    with lock:
        data = list(latest_students_data)
    return jsonify({"students": data, "count": len(data)})

@app.route('/api/register', methods=['POST'])
def register_student():
    """Registers a new face from the current live frame."""
    global pipeline, latest_annotated_frame, lock
    
    data = request.json
    name = data.get('name')
    
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
        
    with lock:
        if latest_annotated_frame is None:
            return jsonify({"success": False, "error": "No camera frame available"}), 500
        frame_to_register = latest_annotated_frame.copy()
        
    print(f"API Registration attempt for: {name}")
    success = pipeline.register_face(frame_to_register, name)
    
    if success:
        return jsonify({"success": True, "message": f"Successfully registered {name}"})
    else:
        return jsonify({"success": False, "error": "No face detected on screen to register"}), 400

@app.route('/api/status')
def get_status():
    """Returns system health status."""
    return jsonify({
        "status": "online",
        "pipeline_active": pipeline is not None
    })

if __name__ == '__main__':
    # Start the webcam pipeline in a daemon thread so it exits when Flask stops
    t = threading.Thread(target=start_pipeline, daemon=True)
    t.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
