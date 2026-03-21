# # FILE: main.py
# import cv2
# import time
# import numpy as np

# from src.core.core_types import FrameContext
# from src.models.yolo_detector import YoloFaceDetector
# from src.models.retina_detector import RetinaFaceDetector
# from src.processors.emotion_analyzer import EmotionAnalyzer # <-- NEW IMPORT

# def main():
#     config = {
#         "model_path": "weights/yolov8n_100e.pt",
#         "confidence_threshold": 0.5
#     }

#     print("Loading AI Models...")
#     yolo_detector = YoloFaceDetector(config)
#     retina_detector = RetinaFaceDetector(config)
#     emotion_analyzer = EmotionAnalyzer() # <-- INSTANTIATE PROCESSOR

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     print("Starting video stream. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         context = FrameContext(frame=frame.copy(), timestamp=time.time())

#         # --- MODULE 1: Face Detection ---
#         context.students = yolo_detector.detect(context.frame)
#         detector_used = "YOLOv8"

#         if len(context.students) == 0:
#             context.students = retina_detector.detect(context.frame)
#             detector_used = "RetinaFace"

#         # --- MODULE 2: Emotion & Engagement Processing ---
#         # Pass the frame and the detected students to the analyzer
#         if len(context.students) > 0:
#             context.students = emotion_analyzer.process(context.frame, context.students)

#         # --- VISUALIZATION ---
#         for student in context.students:
#             x1, y1, x2, y2 = student.bounding_box
#             box_color = (0, 255, 0) if detector_used == "YOLOv8" else (0, 165, 255)
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
#             # Format the text to display Engagement State
#             display_text = f"{student.engagement_state} ({student.fer_emotion})"
            
#             # Draw Engagement Text
#             cv2.putText(frame, display_text, (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
#             # Draw Landmarks (if RetinaFace triggered)
#             if student.landmarks:
#                 for (x, y) in student.landmarks:
#                     cv2.circle(frame, (x, y), 3, (0, 0, 255), -1) 

#         cv2.imshow("AI Class Engagement - Phase 2", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# FILE: backend/main.py
import os
import cv2
import time

from pipeline import ClassEngagementPipeline

def main():
    # Ensure database folder exists before starting
    os.makedirs("database", exist_ok=True)

    pipeline = ClassEngagementPipeline()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n" + "="*50)
    print("SYSTEM READY")
    print("Press 'q' to quit.")
    print("Press 'r' to REGISTER the current face.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame through the AI pipeline
        context = pipeline.process_frame(frame)
        
        # Draw bounding boxes and text
        annotated_frame = pipeline.annotate_frame(context)

        cv2.imshow("AI Class Engagement", annotated_frame)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            break
            
        # Register Face
        elif key == ord('r'):
            print("\n--- REGISTRATION MODE ---")
            name = input("Enter student name: ")
            
            if name.strip():
                success = pipeline.register_face(frame, name.strip())
                if success:
                    print(f"SUCCESS: Registered '{name.strip()}'")
                else:
                    print("Registration failed: No face detected on screen.")
            else:
                print("Registration cancelled: No name entered.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()