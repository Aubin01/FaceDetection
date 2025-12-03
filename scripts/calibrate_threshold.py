"""
Calibrate threshold on real-world images for better generalization.
This helps reduce false positives when using the model in production.
"""
import argparse
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.datasets import PairDataset, load_manifest
from src.data.transforms import build_transforms
from src.models.backbone import FaceEncoder
from src.trainer import cosine_distance
from src.utils.device import get_device
from src.utils.logging import info
from torch.utils.data import DataLoader


def find_optimal_threshold(distances, labels, target_metric="balanced"):
    """
    Find threshold that optimizes for:
    - 'balanced': F1 score (balance precision/recall)
    - 'precision': High precision (fewer false positives)
    - 'recall': High recall (fewer false negatives)
    """
    thresholds = np.linspace(0.2, 0.8, 100)
    best_score = 0.0
    best_thr = 0.5
    
    for thr in thresholds:
        preds = (distances <= thr).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        
        if target_metric == "balanced":
            # F1 score
            score = 2 * (prec * rec) / (prec + rec + 1e-8)
        elif target_metric == "precision":
            # Optimize precision with minimum 70% recall
            score = prec if rec >= 0.7 else 0
        elif target_metric == "recall":
            # Optimize recall with minimum 70% precision
            score = rec if prec >= 0.7 else 0
        else:
            score = acc
            
        if score > best_score:
            best_score = score
            best_thr = thr
            
    return best_thr, best_score


def main():
    parser = argparse.ArgumentParser(description="Calibrate threshold for better real-world performance")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--target", default="precision", choices=["balanced", "precision", "recall"])
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    processed_root = Path(cfg["paths"]["processed_root"])
    manifest = load_manifest(processed_root)
    _, eval_tfms = build_transforms(cfg["data"]["image_size"], cfg["data"]["augment"])

    identities = manifest["val_identities"] if args.split == "val" else manifest["test_identities"]
    loader = DataLoader(
        PairDataset(processed_root, identities=identities, transform=eval_tfms, max_pairs_per_identity=10),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = FaceEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        num_classes=len(manifest["train_identities"]),
        pretrained=cfg["model"]["pretrained"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    distances, labels = [], []
    with torch.no_grad():
        for img_a, img_b, label in loader:
            emb_a = model.encode(img_a.to(device))
            emb_b = model.encode(img_b.to(device))
            dist = cosine_distance(emb_a, emb_b)
            distances.append(dist.cpu())
            labels.append(label)

    distances = torch.cat(distances).numpy()
    labels = torch.cat(labels).numpy()

    # Find optimal threshold
    info(f"\nCalibrating threshold for target metric: {args.target}")
    old_threshold = ckpt.get("best_threshold", 0.5)
    new_threshold, score = find_optimal_threshold(distances, labels, target_metric=args.target)
    
    # Evaluate with both thresholds
    old_preds = (distances <= old_threshold).astype(int)
    new_preds = (distances <= new_threshold).astype(int)
    
    old_acc = accuracy_score(labels, old_preds)
    new_acc = accuracy_score(labels, new_preds)
    old_prec = precision_score(labels, old_preds, zero_division=0)
    new_prec = precision_score(labels, new_preds, zero_division=0)
    old_rec = recall_score(labels, old_preds, zero_division=0)
    new_rec = recall_score(labels, new_preds, zero_division=0)
    
    info("\n" + "="*60)
    info("THRESHOLD COMPARISON")
    info("="*60)
    info(f"Old Threshold: {old_threshold:.4f}")
    info(f"  Accuracy:  {old_acc:.4f}")
    info(f"  Precision: {old_prec:.4f}")
    info(f"  Recall:    {old_rec:.4f}")
    info(f"\nNew Threshold: {new_threshold:.4f} (optimized for {args.target})")
    info(f"  Accuracy:  {new_acc:.4f}")
    info(f"  Precision: {new_prec:.4f}")
    info(f"  Recall:    {new_rec:.4f}")
    info("="*60)
    
    info(f"\n💡 Recommendation: Use threshold = {new_threshold:.4f} in your UI")
    info(f"   Update UI/app.py: best_threshold = {new_threshold:.4f}")


if __name__ == "__main__":
    main()
