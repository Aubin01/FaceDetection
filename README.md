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
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python src/train/preparedata.py
```
This downloads the LFW dataset via Kagglehub, aligns faces with MTCNN, and creates a training manifest (splits.json) with all identities.

**Note**: The `split/` folder contains the **official LFW pairs files** for standardized evaluation:
- `pairsDevTrain.txt` - 1,100 pairs for validation during training
- `pairsDevTest.txt` - 500 pairs for intermediate testing
- `pairs.txt` - 3,000 pairs (10-fold) for final benchmark evaluation

### 3. Train the Model

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python src/train/train.py
```
- **Stage 1**: Classification learning on all 1,680 LFW identities (6 epochs)
- **Stage 2**: Triplet loss fine-tuning with validation on official pairs (10 epochs)
- Saves best model to: `outputs/checkpoints/best_metric.pt`

### 4. Evaluate the Model

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
python src/evaluate/evaluate.py --checkpoint outputs/checkpoints/best_metric.pt --config src/model/config.yaml --split test
```
- `--split val`: Uses `pairsDevTest.txt` (500 pairs)
- `--split test`: Uses `pairs.txt` (3,000 pairs - official benchmark)

## Web Interface (UI)

### Run the Web App
```bash
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

Edit `src/model/config.yaml` to change:
- Image size
- Training epochs
## Project Structure
- `src/model/` - Model architecture, loss functions, and config.yaml
- `src/train/` - Data preparation and training pipeline
- `src/evaluate/` - Evaluation on official LFW pairs
- `src/UI_demo/` - Web interface templates and static files
- `app.py` - Flask web application (root directory)
- `split/` - Official LFW pairs files (tracked in git for reproducibility)
- `data/` - Downloaded and processed dataset (ignored by git)
- `outputs/` - Trained models and evaluation results (ignored by git)
- `app.py` - Flask web application

## Requirements
- Python 3.8+
- PyTorch
- CUDA GPU recommended (but CPU works, just slower)
