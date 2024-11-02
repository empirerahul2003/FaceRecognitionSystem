from deepface import DeepFace
import cv2
import os
import numpy as np
import time
import threading

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir='known_faces'):
        print("\nInitializing Face Recognition System...")
        self.known_faces_dir = known_faces_dir
        self.known_faces = []
        self.model_name = "VGG-Face"
        self.current_detections = []  # Store multiple detections
        self.processing_frame = False
        self.load_known_faces()
        self.detection_threshold = 0.6  # Adjust this threshold for detection sensitivity

    def load_known_faces(self):
        print("Loading known faces...")
        face_count = 0
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.known_faces_dir, filename)
                self.known_faces.append({
                    'path': image_path,
                    'name': os.path.splitext(filename)[0]
                })
                face_count += 1
                print(f"Loaded face for: {os.path.splitext(filename)[0]}")
        print(f"Total faces loaded: {face_count}")

    def process_frame_async(self, frame):
        """Process frame in background thread"""
        if self.processing_frame:
            return

        self.processing_frame = True
        thread = threading.Thread(target=self._process_frame_thread, args=(frame.copy(),))
        thread.daemon = True
        thread.start()

    def _process_frame_thread(self, frame):
        """Background thread for face recognition"""
        try:
            # Save frame temporarily
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            # Extract all faces from the frame first
            faces = DeepFace.extract_faces(temp_frame_path, enforce_detection=False)
            current_detections = []
            
            # Process each detected face
            for face in faces:
                facial_area = face['facial_area']
                face_detected = False
                
                # Create a temporary crop of the face
                face_crop = frame[
                    facial_area['y']:facial_area['y'] + facial_area['h'],
                    facial_area['x']:facial_area['x'] + facial_area['w']
                ]
                face_crop_path = "temp_face.jpg"
                cv2.imwrite(face_crop_path, face_crop)
                
                # Compare with each known face
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
                
                # If face not recognized, mark as Unknown
                if not face_detected:
                    current_detections.append({
                        'x': facial_area['x'],
                        'y': facial_area['y'],
                        'w': facial_area['w'],
                        'h': facial_area['h'],
                        'name': 'Unknown',
                        'confidence': 0
                    })
                
                # Clean up temporary face crop
                if os.path.exists(face_crop_path):
                    os.remove(face_crop_path)
            
            # Update current detections
            self.current_detections = current_detections
            
            # Clean up temporary frame
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
                
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
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

class DroidCamStream:
    def __init__(self):
        self.ip = "192.168.1.100"  # Change this to your phone's IP
        self.port = "4747"
        self.camera_url = f"http://{self.ip}:{self.port}/video"
        self.cap = None

    def setup_camera(self):
        print("\n=== DroidCam Setup ===")
        print("Current settings:")
        print(f"IP: {self.ip}")
        print(f"Port: {self.port}")
        
        change = input("\nDo you want to change these settings? (y/n): ").lower()
        if change == 'y':
            self.ip = input("Enter IP address (e.g., 192.168.1.100): ")
            self.port = input("Enter port (default is 4747): ")
            self.camera_url = f"http://{self.ip}:{self.port}/video"

    def connect(self):
        print(f"\nConnecting to: {self.camera_url}")
        self.cap = cv2.VideoCapture(self.camera_url)
        
        # High quality settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("Error: Could not connect to DroidCam")
            return False
        print("Successfully connected to DroidCam!")
        return True

def main():
    print("=== Face Recognition with DroidCam ===")
    print("Make sure:")
    print("1. DroidCam app is running on your phone")
    print("2. Phone and PC are on the same network")
    print("3. You know the IP address shown in DroidCam app")
    print("4. Known faces are in the 'known_faces' directory")
    
    face_system = FaceRecognitionSystem()
    if not face_system.known_faces:
        print("No faces were loaded. Exiting...")
        return

    droid_cam = DroidCamStream()
    droid_cam.setup_camera()
    
    if not droid_cam.connect():
        print("\nTroubleshooting tips:")
        print("1. Check if IP address is correct")
        print("2. Make sure DroidCam app is running")
        print("3. Verify both devices are on same network")
        return

    print("\nStarting face recognition...")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    frame_count = 0
    start_time = time.time()
    process_interval = 10  # Process every 10 frames
    
    # Create resizable window
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    
    # Optional: Set initial window size
    cv2.resizeWindow('Face Recognition', 1280, 720)

    while True:
        ret, frame = droid_cam.cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Process frame in background every N frames
        if frame_count % process_interval == 0:
            face_system.process_frame_async(frame)

        # Draw all current detections
        face_system.draw_detections(frame)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display total faces detected
            cv2.putText(frame, f"Faces: {len(face_system.current_detections)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame in resizable window
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
        elif key == ord('f'):  # Add fullscreen toggle
            if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Cleanup
    droid_cam.cap.release()
    cv2.destroyAllWindows()
    print("\nProgram ended")
    if elapsed_time > 0:
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    main()