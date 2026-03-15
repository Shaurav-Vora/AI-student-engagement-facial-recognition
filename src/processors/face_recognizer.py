# FILE: src/processors/face_recognizer.py
import os
import cv2
import numpy as np
from deepface import DeepFace
from typing import List

from src.core.core_types import StudentState

class FaceRecognizer:
    def __init__(self, db_path="database"):
        self.db_path = db_path
        print(f"Initializing Face Recognizer with database at: {self.db_path}")

    def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
        """Iterates through students, crops face, and identifies them against the database."""
        for student in students:
            x1, y1, x2, y2 = student.bounding_box
            
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]

            try:
                # DeepFace searches the db_path for a match
                dfs = DeepFace.find(
                    img_path=face_crop, 
                    db_path=self.db_path, 
                    enforce_detection=False, 
                    silent=True,
                    align=False,
                    model_name="OpenFace"
                )
                
                if len(dfs) > 0 and not dfs[0].empty:
                    matched_path = dfs[0].iloc[0]['identity']
                    # Extract just the name from the file path
                    name = os.path.splitext(os.path.basename(matched_path))[0]
                    student.name = name
                else:
                    student.name = "Unknown"
            except Exception:
                student.name = "Unknown"
                
        return students