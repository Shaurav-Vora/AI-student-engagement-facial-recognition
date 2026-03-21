# FILE: src/processors/face_recognizer.py
import os
import cv2
import numpy as np
from typing import List

from src.core.core_types import StudentState

class FaceRecognizer:
    def __init__(self, db_path="database", model_path="weights/face_recognition_sface_2021dec.onnx", threshold=0.363):
        self.db_path = db_path
        self.threshold = threshold  # Cosine similarity threshold for SFace

        print(f"Loading SFace recognition model from {model_path}...")
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")

        # Load all known face embeddings from database on startup
        self.known_names = []
        self.known_embeddings = []
        self._load_database()

    def _load_database(self):
        """Load all .npy embedding files from the database folder."""
        self.known_names = []
        self.known_embeddings = []

        if not os.path.exists(self.db_path):
            return

        for filename in os.listdir(self.db_path):
            if filename.endswith(".npy"):
                name = os.path.splitext(filename)[0]
                embedding = np.load(os.path.join(self.db_path, filename))
                self.known_names.append(name)
                self.known_embeddings.append(embedding)

        print(f"  Loaded {len(self.known_names)} known face(s): {self.known_names}")

    def register(self, frame: np.ndarray, bounding_box, name: str):
        """Register a new face: compute its embedding and save to database."""
        x1, y1, x2, y2 = bounding_box

        # SFace expects the face to be aligned using a specific format
        # We need to provide face info as a 1x15 array for alignCrop
        # Since we already have the crop coordinates, we do manual alignment
        face_crop = frame[y1:y2, x1:x2]
        aligned = cv2.resize(face_crop, (112, 112))

        # Get the 128-d embedding
        embedding = self.recognizer.feature(aligned)

        # Save embedding and image
        np.save(os.path.join(self.db_path, f"{name}.npy"), embedding)
        cv2.imwrite(os.path.join(self.db_path, f"{name}.jpg"), face_crop)

        # Reload database to include the new face
        self._load_database()

    def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
        """Iterates through students, computes embeddings, and identifies them."""
        if len(self.known_embeddings) == 0:
            return students

        for student in students:
            x1, y1, x2, y2 = student.bounding_box

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop and resize to 112x112 (SFace input size)
            face_crop = frame[y1:y2, x1:x2]
            aligned = cv2.resize(face_crop, (112, 112))

            # Get embedding for the current face
            embedding = self.recognizer.feature(aligned)

            # Compare against all known faces using cosine similarity
            best_score = -1.0
            best_name = "Unknown"

            for known_name, known_emb in zip(self.known_names, self.known_embeddings):
                score = self.recognizer.match(embedding, known_emb, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = known_name

            # Only assign name if similarity exceeds threshold
            if best_score >= self.threshold:
                student.name = best_name
            else:
                student.name = "Unknown"

        return students