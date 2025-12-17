# Description: Evaluate a trained face model on LFW verification pairs.
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.model.model import FaceEncoder, cosine_distance
from src.train.preparedata import LFWPairsDataset, build_transforms, load_manifest


def get_device() -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(checkpoint: Path, config_path: Path, split: str, threshold_override: float | None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    processed_root = Path(cfg["paths"]["processed_root"])
    manifest = load_manifest(processed_root)
    _, eval_tfms = build_transforms(cfg["data"]["image_size"], cfg["data"]["augment"])

    # Select official LFW pairs file
    split_root = Path(cfg["paths"].get("split_root", "split"))
    pairs_mapping = {
        "val": "pairsDevTest.txt",    # 500 pairs - intermediate validation
        "test": "pairs.txt"            # 3,000 pairs - official 10-fold benchmark
    }
    pairs_file = split_root / pairs_mapping[split]
    
    loader = DataLoader(
        LFWPairsDataset(processed_root, pairs_file=pairs_file, transform=eval_tfms),
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

    if threshold_override is None:
        threshold = ckpt.get("best_threshold") or find_best_threshold(distances, labels)[0]
    else:
        threshold = threshold_override

    preds = (distances <= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"Split={split} threshold={threshold:.3f} accuracy={acc:.4f}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

    eval_dir = Path(cfg["paths"]["outputs"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{split}_results.csv"

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Checkpoint", checkpoint])
        writer.writerow(["Split", split])
        writer.writerow(["Threshold", f"{threshold:.4f}"])
        writer.writerow(["Accuracy", f"{acc:.4f}"])
        writer.writerow(["Precision", f"{precision:.4f}"])
        writer.writerow(["Recall", f"{recall:.4f}"])
        writer.writerow(["F1-Score", f"{f1:.4f}"])
        writer.writerow(["True Positives", tp])
        writer.writerow(["True Negatives", tn])
        writer.writerow(["False Positives", fp])
        writer.writerow(["False Negatives", fn])
        writer.writerow(["Total Pairs", len(labels)])
        writer.writerow(["Same Person Pairs", int(labels.sum())])
        writer.writerow(["Different Person Pairs", int((1 - labels).sum())])

    print(f"Results saved to: {csv_path}")

    detailed_csv_path = eval_dir / f"{split}_results_detailed.csv"
    with open(detailed_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Pair_Index", "True_Label", "Predicted_Label", "Distance", "Correct"])
        for idx, (label, pred, dist) in enumerate(zip(labels, preds, distances)):
            writer.writerow([idx, int(label), int(pred), f"{dist:.6f}", label == pred])

    print(f"Detailed predictions saved to: {detailed_csv_path}")


def find_best_threshold(distances: np.ndarray, labels: np.ndarray):
    thresholds = np.linspace(0.3, 1.2, 50)
    best_acc, best_thr = 0.0, 0.7
    for thr in thresholds:
        preds = (distances <= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained face model on LFW pairs.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_metric.pt", type=str, help="Path to saved checkpoint")
    parser.add_argument("--config", default="src/model/config.yaml", type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", default=None, type=float)
    args = parser.parse_args()

    evaluate_model(
        checkpoint=Path(args.checkpoint),
        config_path=Path(args.config),
        split=args.split,
        threshold_override=args.threshold,
    )


if __name__ == "__main__":
    main()
