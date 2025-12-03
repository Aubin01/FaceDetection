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
This downloads the LFW dataset, detects faces, and creates train/val/test splits by identity.

### 3. Train the Model

**Option A: Standard training (recommended for general use)**
```bash
python scripts/train.py --config configs/default.yaml
```
- Trains on all identities (classification + triplet loss)
- Two stages: classification pretraining (6 epochs) + triplet fine-tuning (10 epochs)
- Better for generalization to new faces
- Saves to: `outputs/checkpoints/best_metric.pt`

**Option B: Train on official pairs (for maximum benchmark accuracy)**
```bash
# First, place pairsDevTrain.txt in project root
python scripts/train_official_pairs.py --config configs/default.yaml --epochs 50
```
- Trains directly on verification pairs (contrastive loss)
- May score higher on LFW benchmark
- Less generalizable to faces outside LFW
- Saves to: `outputs/checkpoints/best_pairs_trained.pt`

**Compare both**: Train with both methods and test which performs better on your use case!

### 4. Evaluate the Model

**Option A: Official LFW benchmark (for comparing with research papers)**
```bash
# First, place pairsDevTest.txt in project root
python scripts/evaluate_official.py --checkpoint outputs/checkpoints/best_metric.pt --split test
```
Download `pairsDevTest.txt` from: http://vis-www.cs.umass.edu/lfw/

**Option B: Quick validation (custom test set)**
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
