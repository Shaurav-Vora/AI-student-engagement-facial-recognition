# FILE: src/processors/emotion_analyzer.py
import cv2
import numpy as np
from fer.fer import FER
from typing import List
from collections import deque, Counter

from src.core.core_types import StudentState

class EmotionAnalyzer:
    def __init__(self, smoothing_window=10):
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

        # Temporal smoothing: stores recent predictions per face index
        # Uses a majority vote over the last N frames to stabilize output
        self.smoothing_window = smoothing_window
        self.history = {}  # key: face index -> deque of recent emotions
        self.failure_count = {}  # key: face index -> consecutive FER failure count

    def _get_smoothed_emotion(self, face_id: int, raw_emotion: str) -> str:
        """Return the most common emotion over the last N frames for this face."""
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.smoothing_window)
        
        self.history[face_id].append(raw_emotion)
        
        # Majority vote
        counts = Counter(self.history[face_id])
        return counts.most_common(1)[0][0]

    def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
        h, w = frame.shape[:2]
        
        for idx, student in enumerate(students):
            x1, y1, x2, y2 = student.bounding_box
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Add 30% padding around the face for better emotion recognition
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
                result = self.detector.detect_emotions(face_crop)
                if result and len(result) > 0:
                    emotions = result[0]["emotions"]  # dict of emotion -> score
                else:
                    emotions = None
            except Exception:
                emotions = None

            if emotions:
                # Reset failure counter on successful detection
                self.failure_count[idx] = 0
                
                # Get the top emotion normally
                top_emotion = max(emotions, key=emotions.get)
                
                # Smart override: if "neutral" wins, check if negative emotions
                # collectively suggest struggling/disengagement
                negative_emotions = ["sad", "angry", "disgust", "fear"]
                negative_sum = sum(emotions.get(e, 0) for e in negative_emotions)
                
                if top_emotion == "neutral" and negative_sum >= 0.3:
                    # Pick the strongest negative emotion instead
                    top_emotion = max(negative_emotions, key=lambda e: emotions.get(e, 0))
                
                smoothed = self._get_smoothed_emotion(idx, top_emotion)
                student.fer_emotion = smoothed
                student.engagement_state = self.engagement_mapping.get(smoothed, "Unknown")
            else:
                # Face detected but emotion unreadable (hand occlusion, head down, etc.)
                # This is itself a signal of disengagement
                if idx not in self.failure_count:
                    self.failure_count[idx] = 0
                self.failure_count[idx] += 1
                
                if self.failure_count[idx] >= 3:
                    # Persistent occlusion = likely disengaged (head on desk, face in hands)
                    smoothed = self._get_smoothed_emotion(idx, "sad")
                    student.fer_emotion = "occluded"
                    student.engagement_state = "Disengaged / Distressed"
                else:
                    # Brief occlusion — use last known smoothed emotion as fallback
                    if idx in self.history and len(self.history[idx]) > 0:
                        counts = Counter(self.history[idx])
                        last_emotion = counts.most_common(1)[0][0]
                        student.fer_emotion = last_emotion
                        student.engagement_state = self.engagement_mapping.get(last_emotion, "Unknown")
                    else:
                        student.fer_emotion = "None"
                        student.engagement_state = "Unknown"
                
        return students