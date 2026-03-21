from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.core.core_types import StudentState

class BaseFaceDetector(ABC):
    """
    Abstract base class for all face detection models.
    Ensures that any model we plug into the pipeline behaves the exact same way.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Logic to load the model weights (e.g., .pt file or .engine file).
        Must be implemented by the child class.
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[StudentState]:
        """
        Takes an image, runs inference, and returns a list of StudentState objects.
        Must be implemented by the child class.
        """
        pass