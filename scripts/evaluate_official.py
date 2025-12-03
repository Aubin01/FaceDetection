"""
Evaluate using official LFW pairs protocol for benchmark comparison.
This uses the standard LFW pairs files instead of random splits.
"""
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

from src.data.lfw_pairs import LFWPairsDataset, get_lfw_pairs_path
from src.data.transforms import build_transforms
from src.models.backbone import FaceEncoder
from src.trainer import cosine_distance, find_best_threshold
from src.utils.device import get_device
from src.utils.logging import info
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate on official LFW pairs (benchmark protocol)")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to saved checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--pairs-dir", default=None, type=str, help="Directory containing pairs files")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    processed_root = Path(cfg["paths"]["processed_root"])
    
    # Get pairs file
    pairs_dir = Path(args.pairs_dir) if args.pairs_dir else ROOT
    try:
        pairs_file = get_lfw_pairs_path(pairs_dir, args.split)
        info(f"Using official LFW pairs file: {pairs_file}")
    except FileNotFoundError as e:
        info(f"Error: {e}")
        info(f"\nPlease download the official LFW pairs files:")
        info(f"  pairsDevTrain.txt and pairsDevTest.txt")
        info(f"  from: http://vis-www.cs.umass.edu/lfw/")
        info(f"  and place them in: {ROOT}")
        return

    _, eval_tfms = build_transforms(cfg["data"]["image_size"], cfg["data"]["augment"])

    # Load dataset with official pairs
    dataset = LFWPairsDataset(
        data_root=processed_root,
        pairs_file=pairs_file,
        transform=eval_tfms
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    # Load model
    # For official evaluation, we don't need num_classes (no training identities)
    model = FaceEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        num_classes=None,  # No classifier needed for evaluation
        pretrained=cfg["model"]["pretrained"],
    )
    
    # Load weights (ignore classifier if present)
    state_dict = ckpt["model_state_dict"]
    # Remove classifier weights if they exist
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    info(f"\nEvaluating on {len(dataset)} official LFW pairs ({args.split} split)")
    
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
    threshold, _ = find_best_threshold(distances, labels)
    preds = (distances <= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    info("\n" + "="*60)
    info(f"OFFICIAL LFW BENCHMARK RESULTS ({args.split.upper()} SPLIT)")
    info("="*60)
    info(f"Threshold: {threshold:.4f}")
    info(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    info(f"Precision: {precision:.4f}")
    info(f"Recall:    {recall:.4f}")
    info(f"F1-Score:  {f1:.4f}")
    info(f"\nConfusion Matrix:")
    info(f"  True Positives:  {tp}")
    info(f"  True Negatives:  {tn}")
    info(f"  False Positives: {fp}")
    info(f"  False Negatives: {fn}")
    info("="*60)
    
    # Save results
    eval_dir = Path(cfg["paths"]["outputs"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    csv_path = eval_dir / f"official_lfw_{checkpoint_name}_{args.split}_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Protocol', 'Official LFW Benchmark'])
        writer.writerow(['Checkpoint', args.checkpoint])
        writer.writerow(['Split', args.split])
        writer.writerow(['Pairs File', str(pairs_file)])
        writer.writerow(['Total Pairs', len(labels)])
        writer.writerow(['Threshold', f'{threshold:.4f}'])
        writer.writerow(['Accuracy', f'{acc:.4f}'])
        writer.writerow(['Precision', f'{precision:.4f}'])
        writer.writerow(['Recall', f'{recall:.4f}'])
        writer.writerow(['F1-Score', f'{f1:.4f}'])
        writer.writerow(['True Positives', tp])
        writer.writerow(['True Negatives', tn])
        writer.writerow(['False Positives', fp])
        writer.writerow(['False Negatives', fn])
    
    info(f"\n✓ Results saved to: {csv_path}")
    info(f"\nNote: Official LFW benchmark typically reports mean accuracy across 10 folds.")
    info(f"This evaluation uses a single split. For full benchmark, evaluate on all folds.")


if __name__ == "__main__":
    main()
