# FILE: src/processors/emotion_analyzer.py
import cv2
import numpy as np
from fer.fer import FER
from typing import List

from src.core.core_types import StudentState

class EmotionAnalyzer:
    def __init__(self):
        print("Loading FER Emotion model...")
        # mtcnn=False because YuNet already found the face for us!
        self.detector = FER(mtcnn=False) 
        
        self.engagement_mapping = {
            "happy": "Engaged / Positive",
            "surprise": "Engaged / Positive",
            "neutral": "Passively Engaged",
            "disgust": "Struggling",
            "angry": "Struggling", 
            "sad": "Disengaged / Distressed",
            "fear": "Disengaged / Distressed"
        }

    def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
        for student in students:
            x1, y1, x2, y2 = student.bounding_box
            
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            
            try:
                emotion, score = self.detector.top_emotion(face_crop)
            except Exception:
                emotion = None

            if emotion:
                student.fer_emotion = emotion
                student.engagement_state = self.engagement_mapping.get(emotion, "Unknown")
            else:
                student.fer_emotion = "None"
                student.engagement_state = "Unknown"
                
        return students