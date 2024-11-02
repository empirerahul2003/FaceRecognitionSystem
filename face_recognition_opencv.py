from deepface import DeepFace
import cv2
import os
import numpy as np
import time
import threading
from tqdm import tqdm
import sys

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir='known_faces'):
        print("\nInitializing Face Recognition System...")
        self.known_faces_dir = known_faces_dir
        self.known_faces = []
        self.model_name = "VGG-Face"
        self.current_detections = []
        self.processing_frame = False
        self.load_known_faces()
        self.detection_threshold = 0.6

    def load_known_faces(self):
        """Load known faces with progress bar"""
        print("Loading known faces...")
        face_files = [f for f in os.listdir(self.known_faces_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Use tqdm for progress bar
        for filename in tqdm(face_files, desc="Loading faces"):
            image_path = os.path.join(self.known_faces_dir, filename)
            self.known_faces.append({
                'path': image_path,
                'name': os.path.splitext(filename)[0]
            })
            print(f"Loaded face for: {os.path.splitext(filename)[0]}")
        
        print(f"Total faces loaded: {len(self.known_faces)}")

    def process_frame_async(self, frame):
        """Process frame in background thread with progress tracking"""
        if self.processing_frame:
            return

        self.processing_frame = True
        thread = threading.Thread(target=self._process_frame_thread, args=(frame.copy(),))
        thread.daemon = True
        thread.start()

    def _process_frame_thread(self, frame):
        try:
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            faces = DeepFace.extract_faces(temp_frame_path, enforce_detection=False)
            current_detections = []
            
            # Process each face with progress bar
            for face in tqdm(faces, desc="Processing faces", leave=False):
                facial_area = face['facial_area']
                face_detected = False
                
                face_crop = frame[
                    facial_area['y']:facial_area['y'] + facial_area['h'],
                    facial_area['x']:facial_area['x'] + facial_area['w']
                ]
                face_crop_path = "temp_face.jpg"
                cv2.imwrite(face_crop_path, face_crop)
                
                # Compare with known faces
                for known_face in self.known_faces:
                    try:
                        result = DeepFace.verify(
                            img1_path=face_crop_path,
                            img2_path=known_face['path'],
                            model_name=self.model_name,
                            enforce_detection=False,
                            distance_metric="cosine"
                        )
                        
                        if result['verified'] and result['distance'] < self.detection_threshold:
                            current_detections.append({
                                'x': facial_area['x'],
                                'y': facial_area['y'],
                                'w': facial_area['w'],
                                'h': facial_area['h'],
                                'name': known_face['name'],
                                'confidence': (1 - result['distance']) * 100
                            })
                            face_detected = True
                            break
                            
                    except Exception as e:
                        continue
                
                if not face_detected:
                    current_detections.append({
                        'x': facial_area['x'],
                        'y': facial_area['y'],
                        'w': facial_area['w'],
                        'h': facial_area['h'],
                        'name': 'Unknown',
                        'confidence': 0
                    })
                
                if os.path.exists(face_crop_path):
                    os.remove(face_crop_path)
            
            self.current_detections = current_detections
            
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
                
        except Exception as e:
            print(f"Error in face recognition: {str(e)}", file=sys.stderr)
        finally:
            self.processing_frame = False

    def draw_detections(self, frame):
        """Draw all current detections on frame"""
        for detection in self.current_detections:
            x = detection['x']
            y = detection['y']
            w = detection['w']
            h = detection['h']
            
            # Choose color based on recognition
            color = (0, 255, 0) if detection['name'] != 'Unknown' else (0, 0, 255)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw name
            cv2.putText(frame, detection['name'], 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, color, 2)
            
            # Draw confidence for known faces
            if detection['name'] != 'Unknown':
                confidence = f"Confidence: {detection['confidence']:.2f}%"
                cv2.putText(frame, confidence, (x, y+h+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print("=== Face Recognition System ===")
    
    try:
        # Initialize face recognition
        face_system = FaceRecognitionSystem()
        if not face_system.known_faces:
            print("No faces were loaded. Exiting...", file=sys.stderr)
            return

        # Initialize webcam
        print("\nInitializing webcam...")
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Error: Could not open camera", file=sys.stderr)
            return

        # Create window
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', 1280, 720)
        
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        print("- Press 'f' to toggle fullscreen")
        
        frame_count = 0
        start_time = time.time()
        process_interval = 10
        
        with tqdm(desc="Processing video", unit=" frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame", file=sys.stderr)
                    break

                if frame_count % process_interval == 0:
                    face_system.process_frame_async(frame)

                face_system.draw_detections(frame)
                
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Faces: {len(face_system.current_detections)}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Face Recognition', frame)
                pbar.update(1)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"\nScreenshot saved as {filename}")
                elif key == ord('f'):
                    if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    else:
                        cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("\nProgram ended")
        if elapsed_time > 0:
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    main()
