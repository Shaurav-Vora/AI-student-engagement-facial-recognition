# FILE: src/models/yunet_detector.py
import cv2
import numpy as np
from typing import List

from src.interfaces.base_detector import BaseFaceDetector
from src.core.core_types import StudentState

class YuNetDetector(BaseFaceDetector):
    def _load_model(self):
        model_path = self.config.get("model_path", "weights/face_detection_yunet_2023mar.onnx")
        conf_threshold = self.config.get("confidence_threshold", 0.5)
        print(f"Loading YuNet model from {model_path}...")
        self.model = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=(320, 320), score_threshold=conf_threshold
        )
        
    def detect(self, frame: np.ndarray) -> List[StudentState]:
        students = []
        height, width = frame.shape[:2]
        self.model.setInputSize((width, height))
        _, faces = self.model.detect(frame)
        
        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                conf = float(face[-1])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                
                # YuNet gives us the 5 landmarks for free!
                landmarks = [
                    (int(face[4]), int(face[5])), (int(face[6]), int(face[7])),
                    (int(face[8]), int(face[9])), (int(face[10]), int(face[11])),
                    (int(face[12]), int(face[13]))
                ]
                
                students.append(StudentState(
                    bounding_box=(x1, y1, x2, y2), confidence=conf, landmarks=landmarks
                ))
        return students