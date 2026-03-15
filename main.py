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

# FILE: main.py
import os
import cv2
import time
import numpy as np

from src.core.core_types import FrameContext
from src.models.yolo_detector import YoloFaceDetector
from src.models.retina_detector import RetinaFaceDetector
from src.models.yunet_detector import YuNetDetector
from src.processors.emotion_analyzer import EmotionAnalyzer
from src.processors.face_recognizer import FaceRecognizer

def main():
    # Ensure database folder exists before starting
    os.makedirs("database", exist_ok=True)

    config = {
        "model_path": "weights/face_detection_yunet_2023mar.onnx",
        "confidence_threshold": 0.5
    }

    print("Loading AI Models...")
    # yolo_detector = YoloFaceDetector(config)
    # retina_detector = RetinaFaceDetector(config)
    yunet_detector = YuNetDetector(config)
    emotion_analyzer = EmotionAnalyzer()
    face_recognizer = FaceRecognizer(db_path="database")

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

        context = FrameContext(frame=frame.copy(), timestamp=time.time())

        # --- MODULE 1: Face Detection ---
        # context.students = yolo_detector.detect(context.frame)
        # detector_used = "YOLOv8"

        # if len(context.students) == 0:
        #     context.students = retina_detector.detect(context.frame)
        #     detector_used = "RetinaFace"

        context.students = yunet_detector.detect(context.frame)
        detector_used = "YuNet"

        # --- MODULE 2 & IDENTITY: Processing ---
        if len(context.students) > 0:
            context.students = face_recognizer.process(context.frame, context.students)
            context.students = emotion_analyzer.process(context.frame, context.students)

        # --- VISUALIZATION ---
        for student in context.students:
            x1, y1, x2, y2 = student.bounding_box
            box_color = (0, 255, 0) if detector_used == "YOLOv8" else (0, 165, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # --- NEW: 2. Draw the 5-point landmarks ---
            if student.landmarks:
                for (x, y) in student.landmarks:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Display Name and Engagement
            display_text = f"{student.name} - {student.engagement_state} ({student.fer_emotion})"
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("AI Class Engagement", frame)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            break
            
        # Register Face
        elif key == ord('r'):
            if len(context.students) > 0:
                # Grab the first face on screen
                student = context.students[0]
                x1, y1, x2, y2 = student.bounding_box
                
                # Ensure valid crop
                if x2 > x1 and y2 > y1:
                    face_crop = context.frame[y1:y2, x1:x2]
                    
                    # Pause and ask for name in the terminal
                    print("\n--- REGISTRATION MODE ---")
                    name = input("Enter student name: ")
                    
                    if name.strip(): # Make sure they didn't just hit enter
                        # Save the image
                        file_path = f"database/{name}.jpg"
                        cv2.imwrite(file_path, face_crop)
                        print(f"SUCCESS: Saved {name} to {file_path}")
                        
                        # Delete the DeepFace cache so it rebuilds on the next frame
                        pkl_path = "database/representations_vgg_face.pkl"
                        if os.path.exists(pkl_path):
                            os.remove(pkl_path)
                            print("Cleared DeepFace cache. System will lag for a second while it rebuilds...")
                    else:
                        print("Registration cancelled: No name entered.")
            else:
                print("\nRegistration failed: No face detected on screen to register.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()