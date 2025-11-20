import argparse
import sys
from pathlib import Path
import csv
from datetime import datetime

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.datasets import PairDataset, load_manifest
from src.data.transforms import build_transforms
from src.models.backbone import FaceEncoder
from src.trainer import cosine_distance, find_best_threshold
from src.utils.device import get_device
from src.utils.logging import info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained face model on LFW pairs.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to saved checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", default=None, type=float)
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

    if args.threshold is None:
        threshold = ckpt.get("best_threshold") or find_best_threshold(distances, labels)[0]
    else:
        threshold = args.threshold

    preds = (distances <= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    # True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    info(f"Split={args.split} threshold={threshold:.3f} accuracy={acc:.4f}")
    info(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    info(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    
    # Save results to CSV
    eval_dir = Path(cfg["paths"]["outputs"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    csv_path = eval_dir / f"evaluation_{checkpoint_name}_{args.split}_{timestamp}.csv"
    
    # Save summary
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Checkpoint', args.checkpoint])
        writer.writerow(['Split', args.split])
        writer.writerow(['Threshold', f'{threshold:.4f}'])
        writer.writerow(['Accuracy', f'{acc:.4f}'])
        writer.writerow(['Precision', f'{precision:.4f}'])
        writer.writerow(['Recall', f'{recall:.4f}'])
        writer.writerow(['F1-Score', f'{f1:.4f}'])
        writer.writerow(['True Positives', tp])
        writer.writerow(['True Negatives', tn])
        writer.writerow(['False Positives', fp])
        writer.writerow(['False Negatives', fn])
        writer.writerow(['Total Pairs', len(labels)])
        writer.writerow(['Same Person Pairs', int(labels.sum())])
        writer.writerow(['Different Person Pairs', int((1 - labels).sum())])
    
    info(f"Results saved to: {csv_path}")
    
    # Save detailed predictions
    detailed_csv_path = eval_dir / f"predictions_{checkpoint_name}_{args.split}_{timestamp}.csv"
    with open(detailed_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair_Index', 'True_Label', 'Predicted_Label', 'Distance', 'Correct'])
        for idx, (label, pred, dist) in enumerate(zip(labels, preds, distances)):
            writer.writerow([idx, int(label), int(pred), f'{dist:.6f}', label == pred])
    
    info(f"Detailed predictions saved to: {detailed_csv_path}")


if __name__ == "__main__":
    main()
