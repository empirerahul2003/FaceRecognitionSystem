# Face Recognition System

A real-time face recognition system built with Python, OpenCV, and DeepFace that supports multiple face detection, webcam integration, and DroidCam support.

## 🌟 Features

- **Multi-Person Detection**: Detect and recognize multiple faces simultaneously
- **Dual Camera Support**: 
  - 💻 Laptop Webcam
  - 📱 DroidCam (Phone Camera)
- **Real-Time Processing**:
  - Face Detection
  - Face Recognition
  - FPS Counter
- **User Interface**:
  - Progress Tracking
  - Screenshot Capability
  - Fullscreen Mode
- **Face Database**:
  - Easy to add new faces
  - Supports multiple images per person

## 🚀 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/FaceRecognitionSystem.git
cd FaceRecognitionSystem
```

2. **Create Virtual Environment**
```bash
# Create environment
python -m venv tdenv

# Activate environment
# For Windows:
.\tdenv\Scripts\activate
# For Linux/Mac:
source tdenv/bin/activate
```

3. **Install Requirements**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Laptop Camera
```bash
python face_recognition_opencv.py
```

### Phone Camera (DroidCam)
```bash
python face_recognition_droidcam.py
```

### Controls
- `q` : Quit program
- `s` : Save screenshot
- `f` : Toggle fullscreen

## 📁 Project Structure

```
FaceRecognitionSystem/
├── known_faces/           # Face database directory
│   └── .gitkeep
├── examples/              # Example images
│   └── sample.jpg
├── face_recognition_opencv.py    # Laptop camera version
├── face_recognition_droidcam.py  # Phone camera version
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

1. **Adding Face Images**
   - Add person's photo to `known_faces/` directory
   - Name format: `person_name.jpg`
   - Use clear, front-facing photos

2. **Camera Settings**
   ```python
   # In face_recognition_opencv.py or face_recognition_droidcam.py
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   cap.set(cv2.CAP_PROP_FPS, 30)
   ```

## 📋 Requirements

- Python 3.8+
- OpenCV
- DeepFace
- NumPy
- tqdm

### DeepFace
This project uses DeepFace for face recognition. For detailed information about DeepFace:
- [DeepFace Documentation](https://github.com/serengil/deepface/blob/master/README.md)

Key DeepFace features used in this project:
```python
# Face Verification
result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")

# Face Recognition
dfs = DeepFace.find(img_path = "img.jpg", db_path = "known_faces/")

# Facial Analysis
demographies = DeepFace.analyze(img_path = "img.jpg")
```
## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- DeepFace for face recognition
- All contributors and users of this project

## 📞 Contact

Your Name - Rahul Kumar

Project Link: 
