import argparse
import sys
from pathlib import Path

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
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained face model on LFW pairs.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to saved checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", default=None, type=float)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    acc = accuracy_score(labels, preds)
    info(f"Split={args.split} threshold={threshold:.3f} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
