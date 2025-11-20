# Face Recognition UI

State-of-the-art web interface for face recognition testing and visualization.

## Features

- **Face Verification**: Upload two images to verify if they belong to the same person
- **Face Detection**: Detect faces in images with confidence scores
- **Embedding Extraction**: Extract 128D face embeddings and view statistics
- **Live Camera**: Real-time face detection using webcam

## Setup

1. Install dependencies:
```bash
pip install flask flask-cors pillow
```

2. Make sure your model checkpoint exists:
```bash
# The UI looks for: outputs/checkpoints/best_classification.pt
```

3. Run the server:
```bash
cd UI
python app.py
```

4. Open your browser:
```
http://localhost:5000
```

## Technologies

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Face Detection**: MTCNN
- **Face Recognition**: FaceNet (PyTorch)
- **Camera**: WebRTC

## API Endpoints

- `GET /api/status` - Check if model is loaded
- `POST /api/detect` - Detect faces in an image
- `POST /api/embed` - Extract face embeddings
- `POST /api/verify` - Verify if two faces match

## Features Showcase

### 1. Face Verification
- Upload two images
- Get similarity score and confidence
- Visual progress bar
- Same/Different person classification

### 2. Face Detection
- Upload image
- Detect multiple faces
- Confidence scores for each face
- Face count statistics

### 3. Embedding Extraction
- Extract 128D feature vector
- View embedding statistics (mean, std, norm)
- Copy embedding for further use

### 4. Live Camera
- Real-time webcam access
- Capture photos
- Use captured photos for verification/detection

## Notes

- Model uses the best classification checkpoint
- Optimal threshold: ~0.52 (from training)
- Supports JPEG, PNG image formats
- Mobile responsive design
