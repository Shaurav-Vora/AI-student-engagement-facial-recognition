from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np

@dataclass
class StudentState:
    """
    Holds all information about a single detected student in a frame.
    Attributes will be populated as the pipeline progresses through phases.
    """
    # Phase 1: Face Detection
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None # For RetinaFace fallback
    
    # Phase 1: EAR & Head Pose (To be populated by later modules)
    ear_score: Optional[float] = None
    head_pose: Optional[Tuple[float, float, float]] = None # (pitch, yaw, roll)
    
    # Future Phases
    tracking_id: Optional[int] = None
    engagement_score: Optional[float] = None
    fer_emotion: Optional[str] = None
    engagement_state: Optional[str] = None

    name: Optional[str] = "Unknown"

@dataclass
class FrameContext:
    """
    Holds the original frame and the list of students detected within it.
    """
    frame: np.ndarray  # The raw image array from OpenCV
    timestamp: float
    students: List[StudentState] = field(default_factory=list)