## Face Recognition (LFW) — Two-Stage Metric Learning

End-to-end face recognition pipeline using MTCNN alignment, FaceNet encoder, classification pretrain, and triplet fine-tuning. Implements identity-disjoint splits on LFW, balanced pair generation, and verification evaluation with threshold search.

### Project layout
- `configs/default.yaml` — hyperparameters, paths, and augment toggles
- `scripts/prepare_data.py` — download LFW (deepfunneled), run MTCNN alignment, build splits
- `scripts/train.py` — classification pretrain + triplet fine-tune, checkpoints to `outputs/checkpoints`
- `scripts/evaluate.py` — load a checkpoint and score verification accuracy on val/test pairs
- `src/data` — dataset prep, aligned dataset loaders, pair sampling, transforms
- `src/models` — FaceNet-based encoder and custom heads/losses
- `src/trainer.py` — training/eval orchestration
- `requirements.txt` — runtime dependencies

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data preparation
```bash
python scripts/prepare_data.py --config configs/default.yaml
```
- Downloads LFW either via KaggleHub (default) or torchvision, based on `data.source`.
- Runs MTCNN alignment and saves cropped faces to `data/processed/<identity>/*.png`.
- Builds identity-disjoint train/val/test split metadata in `data/processed/splits.json`.

If using KaggleHub, set `data.kaggle_dataset` and (if needed) `data.kaggle_subdir` in the config. Example:
```yaml
data:
  source: kagglehub
  kaggle_dataset: atulanandjha/lfwpeople
  kaggle_subdir: lfw-deepfunneled  # if the dataset structure requires it
```

### Training (2 phases back-to-back)
```bash
python scripts/train.py --config configs/default.yaml
```
- Stage 1: classification pretrain over train identities.
- Stage 2: triplet fine-tune with mixed triplet + lightly weighted CE.
- Tracks best val verification accuracy and saves checkpoints under `outputs/checkpoints/`.

### Evaluate a checkpoint
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_metric.pt --config configs/default.yaml --split test
```
- Uses stored best threshold (or will re-tune if missing) to report verification accuracy.

### Notes and tips
- Configurable image size/augment/optimizer settings live in `configs/default.yaml`.
- Adjust `data.min_images_per_identity` if you want to include people with only one photo (triplet quality may drop).
- Training benefits from GPU/Apple MPS; CPU works but is slower. The code auto-selects the best available device unless overridden in the config.
- If you already have an LFW mirror, point `paths.data_root` to it before running `prepare_data.py`.
