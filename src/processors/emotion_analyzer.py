# FILE: src/processors/emotion_analyzer.py
import cv2
import numpy as np
from deepface import DeepFace
from typing import List

from src.core.core_types import StudentState

class EmotionAnalyzer:
    def __init__(self):
        print("Initializing DeepFace Emotion Processor...")
        
        # DeepFace downloads its weights automatically on the first run.
        # Rule-based mapping from your design document
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
        """Iterates through detected students, crops their face, and predicts emotion."""
        
        for student in students:
            # 1. Get the bounding box from Module 1
            x1, y1, x2, y2 = student.bounding_box
            
            # Prevent zero-area crops if boxes are corrupted
            if x2 <= x1 or y2 <= y1:
                continue

            # 2. Crop the face out of the main frame
            face_crop = frame[y1:y2, x1:x2]
            
            # 3. Predict emotion using DeepFace
            try:
                # silent=True keeps terminal clean
                # enforce_detection=False is required because we are feeding it an already-cropped face
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                
                # DeepFace returns a list of dicts if it thinks there are multiple faces, 
                # but we just want the first one.
                if isinstance(analysis, list):
                    analysis = analysis[0]
                    
                emotion = analysis.get('dominant_emotion')
            except Exception as e:
                emotion = None

            # 4. Map and store the results in the DTO
            if emotion:
                student.fer_emotion = emotion
                student.engagement_state = self.engagement_mapping.get(emotion, "Unknown")
            else:
                student.fer_emotion = "None"
                student.engagement_state = "Unknown"
                
        return students

# FILE: src/processors/emotion_analyzer.py
# import cv2
# import numpy as np
# from fer import FER
# from typing import List

# from src.core.core_types import StudentState

# class EmotionAnalyzer:
#     def __init__(self):
#         print("Loading FER Emotion model...")
#         # mtcnn=False because YuNet already found the face for us!
#         self.detector = FER(mtcnn=False) 
        
#         self.engagement_mapping = {
#             "happy": "Engaged / Positive",
#             "surprise": "Engaged / Positive",
#             "neutral": "Passively Engaged",
#             "disgust": "Struggling",
#             "angry": "Struggling", 
#             "sad": "Disengaged / Distressed",
#             "fear": "Disengaged / Distressed"
#         }

#     def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
#         for student in students:
#             x1, y1, x2, y2 = student.bounding_box
            
#             if x2 <= x1 or y2 <= y1:
#                 continue

#             face_crop = frame[y1:y2, x1:x2]
            
#             try:
#                 emotion, score = self.detector.top_emotion(face_crop)
#             except Exception:
#                 emotion = None

#             if emotion:
#                 student.fer_emotion = emotion
#                 student.engagement_state = self.engagement_mapping.get(emotion, "Unknown")
#             else:
#                 student.fer_emotion = "None"
#                 student.engagement_state = "Unknown"
                
#         return students