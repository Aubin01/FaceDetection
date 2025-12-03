# Face Recognition System

A face recognition system that can identify if two face images are from the same person. Uses FaceNet and MTCNN for face detection and recognition.

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Download and prepare face images:
```bash
python scripts/prepare_data.py --config configs/default.yaml
```
This downloads the LFW dataset, detects faces, and splits them into train/val/test sets.

### 3. Train the Model
Train the face recognition model (takes time, needs GPU for speed):
```bash
python scripts/train.py --config configs/default.yaml
```
The model trains in two stages:
- Stage 1: Learn to classify different people (6 epochs)
- Stage 2: Fine-tune with triplet loss for better matching (10 epochs)

Trained models save to `outputs/checkpoints/`

### 4. Test the Model
Check how well the model works:
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_metric.pt --config configs/default.yaml --split test
```

## Web Interface (UI)

### Run the Web App
```bash
cd UI
pip install -r requirements.txt
python app.py
```

Open your browser and go to: **http://localhost:5000**

### What You Can Do
- **Verify Faces**: Upload two photos and check if they are the same person
- **Detect Faces**: Find faces in an image
- **Get Embeddings**: Extract face features from images

### Important
Make sure you have a trained model at `outputs/checkpoints/best_metric.pt` before starting the web app.

## Configuration

Edit `configs/default.yaml` to change:
- Image size
- Training epochs
- Learning rates
- Data augmentation settings

## Project Structure
- `configs/` - Settings and parameters
- `scripts/` - Main programs (prepare data, train, evaluate)
- `src/` - Core code (models, datasets, training)
- `UI/` - Web interface
- `data/` - Dataset files
- `outputs/` - Trained models and results

## Requirements
- Python 3.8+
- PyTorch
- CUDA GPU recommended (but CPU works, just slower)
