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
        h, w = frame.shape[:2]
        
        for student in students:
            x1, y1, x2, y2 = student.bounding_box
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Add 30% padding around the face for better emotion recognition
            # FER needs forehead, chin, and some surrounding context
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * 0.3)
            pad_y = int(bh * 0.3)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            face_crop = frame[py1:py2, px1:px2]
            
            try:
                emotion, score = self.detector.top_emotion(face_crop)
            except Exception:
                emotion = None
                score = 0.0

            if emotion and score and score >= 0.05:
                student.fer_emotion = emotion
                student.engagement_state = self.engagement_mapping.get(emotion, "Unknown")
            else:
                student.fer_emotion = "None"
                student.engagement_state = "Unknown"
                
        return students