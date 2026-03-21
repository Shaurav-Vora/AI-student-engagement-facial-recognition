# FILE: src/models/retina_detector.py
import numpy as np
from typing import List
from retinaface import RetinaFace

from src.interfaces.base_detector import BaseFaceDetector
from src.core.core_types import StudentState

class RetinaFaceDetector(BaseFaceDetector):
    def _load_model(self):
        """RetinaFace handles its own initialization."""
        print("RetinaFace fallback detector initialized.")
        
    def detect(self, frame: np.ndarray) -> List[StudentState]:
        """Runs RetinaFace inference and maps results to StudentState DTOs."""
        students = []
        conf_threshold = self.config.get("confidence_threshold", 0.5)
        
        height, width = frame.shape[:2]
        predictions = RetinaFace.detect_faces(frame)
        
        if isinstance(predictions, dict):
            for face_key, face_data in predictions.items():
                conf = face_data.get("score", 0.0)
                if conf < conf_threshold:
                    continue
                    
                facial_area = face_data.get("facial_area", [0, 0, 0, 0])
                x1, y1, x2, y2 = facial_area
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # EXTRACT THE 5-POINT LANDMARKS
                landmarks_dict = face_data.get("landmarks", {})
                face_landmarks = []
                if landmarks_dict:
                    # Grab the 5 points and ensure they are integers for drawing
                    face_landmarks = [
                        tuple(map(int, landmarks_dict["right_eye"])),
                        tuple(map(int, landmarks_dict["left_eye"])),
                        tuple(map(int, landmarks_dict["nose"])),
                        tuple(map(int, landmarks_dict["mouth_right"])),
                        tuple(map(int, landmarks_dict["mouth_left"]))
                    ]
                
                # Map to our standard DTO, now including landmarks!
                student = StudentState(
                    bounding_box=(x1, y1, x2, y2),
                    confidence=conf,
                    landmarks=face_landmarks
                )
                students.append(student)
                
        return students