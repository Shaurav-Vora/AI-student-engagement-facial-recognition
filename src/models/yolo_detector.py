# FILE: src/models/yolo_detector.py
import numpy as np
from typing import List
from ultralytics import YOLO

from src.interfaces.base_detector import BaseFaceDetector
from src.core.core_types import StudentState

class YoloFaceDetector(BaseFaceDetector):
    def _load_model(self):
        """Loads the YOLOv8-Face PyTorch model."""
        model_path = self.config.get("model_path", "weights/yolov8n-100e.pt")
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        
    def detect(self, frame: np.ndarray) -> List[StudentState]:
        """Runs YOLOv8 inference and maps results to StudentState DTOs."""
        students = []
        conf_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Get frame dimensions for boundary clamping
        height, width = frame.shape[:2]
        
        # OPTIMIZATION 1: Explicitly set imgsz and disable half-precision
        results = self.model(frame, verbose=False, imgsz=640, half=False)
        
        # Parse the results
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                
                # Ignore detections below our threshold
                if conf < conf_threshold:
                    continue
                
                # Extract coordinates as integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # OPTIMIZATION 2: Clamp coordinates to frame dimensions to prevent future crashes
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Map the inference output to our standardized DTO
                student = StudentState(
                    bounding_box=(x1, y1, x2, y2),
                    confidence=conf
                )
                students.append(student)
                
        return students