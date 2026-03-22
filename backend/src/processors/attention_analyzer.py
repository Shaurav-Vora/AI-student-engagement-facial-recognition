# FILE: backend/src/processors/attention_analyzer.py
import cv2
import time
import numpy as np
import math
import os
from typing import List, Dict
from collections import deque
import logging

try:
    from l2cs import Pipeline as L2CSPipeline
    L2CS_AVAILABLE = True
    print("[AttentionAnalyzer] l2cs imported successfully.")
except Exception as e:
    L2CS_AVAILABLE = False
    print(f"[AttentionAnalyzer] WARNING: l2cs import failed: {e}")
    print("[AttentionAnalyzer] Make sure the venv is activated! Gaze estimation disabled.")

from src.core.core_types import StudentState

# Get absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AttentionAnalyzer:
    """
    Module 3 — Attention Ratio (Eye Gaze & Blink Detection)

    Level 1: EAR (Eye Aspect Ratio) via OpenCV FacemarkLBF (68-point landmarks)
    Level 2: Gaze Estimation via L2CS-Net (full-frame, with IoU matching to YuNet students)
    Attention Score = (Frames_Looking_Forward / Total_Frames) * 100, weighted by EAR
    """

    def __init__(self, history_seconds: int = 60, fps_estimate: int = 10):
        print("[AttentionAnalyzer] Initializing...")

        # --- Level 1: EAR via OpenCV FacemarkLBF ---
        self.facemark = None
        lbf_model_path = os.path.join(BACKEND_DIR, "models", "lbfmodel.yaml")
        if os.path.exists(lbf_model_path):
            try:
                self.facemark = cv2.face.createFacemarkLBF()
                self.facemark.loadModel(lbf_model_path)
                print(f"[AttentionAnalyzer] OpenCV FacemarkLBF loaded from {lbf_model_path}")
            except Exception as e:
                logging.error(f"[AttentionAnalyzer] Failed to load FacemarkLBF: {e}")
                self.facemark = None
        else:
            logging.warning(f"[AttentionAnalyzer] LBF model not found at {lbf_model_path}. EAR disabled.")

        # --- Level 2: L2CS-Net Gaze ---
        self.l2cs_pipeline = None
        if L2CS_AVAILABLE:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weights_path = os.path.join(BACKEND_DIR, "models", "L2CSNet_gaze360.pkl")
            try:
                self.l2cs_pipeline = L2CSPipeline(
                    weights=weights_path,
                    arch='ResNet50',
                    device=device
                )
                print(f"[AttentionAnalyzer] L2CS-Net initialized on {device}")
            except Exception as e:
                logging.error(f"[AttentionAnalyzer] Failed to init L2CS-Net: {e}")
                self.l2cs_pipeline = None

        # --- Temporal history for rolling metrics ---
        self.history_seconds = history_seconds
        self.max_history_len = history_seconds * fps_estimate
        self.student_history: Dict[str, Dict] = {}

        # --- Frame skipping for performance ---
        self.frame_counter = 0
        self.skip_interval = 3  # Run heavy inference every Nth frame
        self.cached_l2cs_results = None
        self.cached_ear_landmarks = {}

        # --- Adaptive gaze baseline calibration ---
        # Auto-calibrate the "Forward" baseline from the first N readings
        self.gaze_baseline_pitch = None
        self.gaze_baseline_yaw = None
        self._calibration_pitches = []
        self._calibration_yaws = []
        self._calibration_samples = 30  # Use first 30 readings to establish baseline
        self._calibrated = False

        print("[AttentionAnalyzer] Ready. Gaze will auto-calibrate from first 30 readings.")

    # ------------------------------------------------------------------ #
    #  EAR helpers (68-point OpenCV landmarks)                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ear_from_68(landmarks_68: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio from 68-point facial landmarks.
        Right eye: points 36-41,  Left eye: points 42-47

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        def _single_ear(pts):
            # pts is 6 points: p1..p6 in order
            p1, p2, p3, p4, p5, p6 = pts
            num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
            den = 2.0 * np.linalg.norm(p1 - p4)
            return num / (den + 1e-6)

        right_eye = landmarks_68[36:42]   # 6 points
        left_eye = landmarks_68[42:48]    # 6 points

        return (_single_ear(right_eye) + _single_ear(left_eye)) / 2.0

    # ------------------------------------------------------------------ #
    #  IoU matching helper                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iou(box_a, box_b):
        """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / (union + 1e-6)

    # ------------------------------------------------------------------ #
    #  History helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_history(self, student_id: str) -> dict:
        if student_id not in self.student_history:
            self.student_history[student_id] = {
                'blinks': deque(maxlen=self.max_history_len),
                'gaze': deque(maxlen=self.max_history_len),
            }
        return self.student_history[student_id]

    # ------------------------------------------------------------------ #
    #  Main process()                                                     #
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray, students: List[StudentState]) -> List[StudentState]:
        if not students:
            return students

        current_time = time.time()
        h_img, w_img = frame.shape[:2]

        # Determine whether this is an inference frame or a skip frame
        self.frame_counter += 1
        run_inference = (self.frame_counter % self.skip_interval == 0)

        if run_inference:
            # ============================================================== #
            #  STEP A — Run L2CS-Net ONCE on the FULL frame                  #
            # ============================================================== #
            l2cs_results = None
            if self.l2cs_pipeline is not None:
                try:
                    l2cs_results = self.l2cs_pipeline.step(frame)   # expects BGR
                except Exception as e:
                    logging.error(f"L2CS full-frame error: {e}")
            self.cached_l2cs_results = l2cs_results

            # ============================================================== #
            #  STEP B — Run OpenCV FacemarkLBF for 68-point landmarks        #
            # ============================================================== #
            all_ear_landmarks = {}
            if self.facemark is not None:
                faces_rect = np.array([
                    [int(s.bounding_box[0]), int(s.bounding_box[1]),
                     int(s.bounding_box[2] - s.bounding_box[0]),
                     int(s.bounding_box[3] - s.bounding_box[1])]
                    for s in students
                ], dtype=np.int32)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    ok, landmarks_list = self.facemark.fit(gray, faces_rect)
                    if ok and landmarks_list is not None:
                        for idx, lm in enumerate(landmarks_list):
                            if lm is not None and len(lm) > 0:
                                all_ear_landmarks[idx] = lm[0]
                except Exception as e:
                    logging.error(f"FacemarkLBF fitting error: {e}")
            self.cached_ear_landmarks = all_ear_landmarks
        else:
            # Reuse cached results from the last inference frame
            l2cs_results = self.cached_l2cs_results
            all_ear_landmarks = self.cached_ear_landmarks

        # ============================================================== #
        #  STEP C — For each student, assign EAR + Gaze + Attention      #
        # ============================================================== #
        for i, student in enumerate(students):
            student_id = student.name if student.name != "Unknown" else f"unk_{id(student)}"
            history = self._get_history(student_id)

            # ---------- Level 1: EAR & Blink ----------
            ear = None
            is_blink = False

            if i in all_ear_landmarks:
                ear = self._ear_from_68(all_ear_landmarks[i])
                student.ear_score = round(float(ear), 3)
                is_blink = ear < 0.25

            history['blinks'].append((current_time, is_blink))

            # Blink rate (rising-edge count over the history window)
            blinks_counted = 0
            prev_b = False
            valid_blinks = [(t, b) for t, b in history['blinks']
                            if current_time - t <= self.history_seconds]
            for _, b in valid_blinks:
                if b and not prev_b:
                    blinks_counted += 1
                prev_b = b
            time_span = max(1.0, current_time - valid_blinks[0][0]) if valid_blinks else 1.0
            time_span = min(self.history_seconds, time_span)
            student.blink_rate = round((blinks_counted / time_span) * 60.0, 1)

            # ---------- Level 2: Gaze (L2CS match by IoU) ----------
            gaze_dir = "N/A"
            is_looking_forward = True

            if l2cs_results is not None and len(l2cs_results.pitch) > 0:
                # Match this student's YuNet bbox to the closest L2CS bbox
                s_box = student.bounding_box  # (x1, y1, x2, y2)
                best_iou = 0.0
                best_idx = -1

                for j in range(len(l2cs_results.bboxes)):
                    l_box = l2cs_results.bboxes[j]  # [x1, y1, x2, y2]
                    overlap = self._iou(s_box, l_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_idx = j

                if best_idx >= 0 and best_iou > 0.2:
                    pitch_rad = float(l2cs_results.pitch[best_idx])
                    yaw_rad = float(l2cs_results.yaw[best_idx])
                    pitch_deg = math.degrees(pitch_rad)
                    yaw_deg = math.degrees(yaw_rad)

                    # --- Auto-calibrate baseline from first N readings ---
                    if not self._calibrated:
                        self._calibration_pitches.append(pitch_deg)
                        self._calibration_yaws.append(yaw_deg)
                        if len(self._calibration_pitches) >= self._calibration_samples:
                            self.gaze_baseline_pitch = float(np.median(self._calibration_pitches))
                            self.gaze_baseline_yaw = float(np.median(self._calibration_yaws))
                            self._calibrated = True
                            print(f"[GAZE] Baseline calibrated: pitch={self.gaze_baseline_pitch:.1f}° yaw={self.gaze_baseline_yaw:.1f}°")
                        gaze_dir = "Calibrating..."
                        is_looking_forward = True
                    else:
                        # Compute deviation from baseline
                        dp = pitch_deg - self.gaze_baseline_pitch
                        dy = yaw_deg - self.gaze_baseline_yaw

                        # Classification by deviation:
                        #   Left:  pitch shifts positive by >25°
                        #   Right: pitch shifts negative by >80°
                        #   Down:  pitch shifts negative by >25° AND yaw shifts negative by >12°
                        #   Up:    yaw shifts negative by >12° (pitch stays moderate)
                        #   Forward: small deviations

                        if dp > 25:
                            gaze_dir = "Sideways"
                            is_looking_forward = False
                        elif dp < -80:
                            gaze_dir = "Sideways"
                            is_looking_forward = False
                        elif dp < -25 and dy < -12:
                            gaze_dir = "Down"
                            is_looking_forward = False
                        elif dp < -25:
                            gaze_dir = "Sideways"
                            is_looking_forward = False
                        elif dy < -12:
                            gaze_dir = "Up"
                            is_looking_forward = True
                        else:
                            gaze_dir = "Forward"
                            is_looking_forward = True

                        print(f"[GAZE] p={pitch_deg:.0f}° y={yaw_deg:.0f}° dp={dp:.0f} dy={dy:.0f} → {gaze_dir}")
                else:
                    gaze_dir = "No Match"
                    is_looking_forward = False

            # If eyes are closed (EAR), override gaze
            if ear is not None and ear < 0.20:
                gaze_dir = "Eyes Closed"
                is_looking_forward = False

            student.gaze_direction = gaze_dir
            history['gaze'].append((current_time, is_looking_forward))

            # ---------- Level 3: Attention Score ----------
            valid_gaze = [(t, gf) for t, gf in history['gaze']
                          if current_time - t <= self.history_seconds]
            if not valid_gaze:
                student.attention_score = 100.0
            else:
                forward_count = sum(1 for _, gf in valid_gaze if gf)
                total_count = len(valid_gaze)
                score = (forward_count / total_count) * 100.0

                # Weighted penalty: if EAR is consistently very low → drowsy
                if ear is not None and ear < 0.22:
                    score *= 0.8

                # Additional penalty if blink rate is very high (fatigue)
                if student.blink_rate and student.blink_rate > 25:
                    score *= 0.9

                student.attention_score = round(max(0.0, min(100.0, score)), 1)

            # Optionally influence engagement state
            if student.engagement_state and student.attention_score < 40.0:
                if "Engaged" in student.engagement_state:
                    student.engagement_state = "Distracted"

        return students
