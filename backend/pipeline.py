# FILE: backend/pipeline.py
import cv2
import time
import numpy as np

import os

from src.core.core_types import FrameContext
from src.models.yunet_detector import YuNetDetector
from src.processors.emotion_analyzer import EmotionAnalyzer
from src.processors.face_recognizer import FaceRecognizer

# Get absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

class ClassEngagementPipeline:
    def __init__(self):
        print("Initializing Engagement Pipeline...")
        config = {
            "model_path": os.path.join(BACKEND_DIR, "weights", "face_detection_yunet_2023mar.onnx"),
            "confidence_threshold": 0.5
        }
        
        db_path = os.path.join(BACKEND_DIR, "database")
        sface_path = os.path.join(BACKEND_DIR, "weights", "face_recognition_sface_2021dec.onnx")
        
        self.yunet_detector = YuNetDetector(config)
        self.emotion_analyzer = EmotionAnalyzer()
        self.face_recognizer = FaceRecognizer(db_path=db_path, model_path=sface_path)

    def process_frame(self, frame: np.ndarray) -> FrameContext:
        """Processes a single frame, updates students list, and returns the context."""
        context = FrameContext(frame=frame.copy(), timestamp=time.time())

        # 1. Face Detection
        context.students = self.yunet_detector.detect(context.frame)
        self.detector_used = "YuNet"

        # 2. Identity & Emotion Processing
        if len(context.students) > 0:
            context.students = self.face_recognizer.process(context.frame, context.students)
            context.students = self.emotion_analyzer.process(context.frame, context.students)

        return context

    def annotate_frame(self, context: FrameContext) -> np.ndarray:
        """Draws bounding boxes, landmarks, and text on the frame."""
        annotated_frame = context.frame.copy()
        
        for student in context.students:
            x1, y1, x2, y2 = student.bounding_box
            box_color = (0, 165, 255) # Orange for YuNet
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw the 5-point landmarks
            if student.landmarks:
                for (x, y) in student.landmarks:
                    cv2.circle(annotated_frame, (x, y), 3, (0, 0, 255), -1)

            # Display Name and Engagement
            display_text = f"{student.name} - {student.engagement_state} ({student.fer_emotion})"
            cv2.putText(annotated_frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        return annotated_frame

    def register_face(self, frame: np.ndarray, name: str) -> bool:
        """Attempts to register the first detected face in the frame."""
        # Run detection just to find the bounding box
        students = self.yunet_detector.detect(frame)
        
        if len(students) > 0:
            student = students[0]
            x1, y1, x2, y2 = student.bounding_box
            
            if x2 > x1 and y2 > y1:
                self.face_recognizer.register(frame, student.bounding_box, name)
                return True
        return False
