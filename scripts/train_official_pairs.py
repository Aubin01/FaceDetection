"""
Train using official LFW pairs protocol.
Uses pairsDevTrain.txt for training and pairsDevTest.txt for validation.

This is different from the standard approach:
- Standard: Train on all identities (classification + triplet)
- This: Train directly on verification pairs (contrastive/triplet loss only)

May give better benchmark scores but less generalizable embeddings.
"""
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.lfw_pairs import LFWPairsDataset, get_lfw_pairs_path
from src.data.transforms import build_transforms
from src.models.backbone import FaceEncoder
from src.trainer import cosine_distance, find_best_threshold
from src.utils.device import get_device
from src.utils.logging import info
from src.utils.seed import set_seed
from sklearn.metrics import accuracy_score


class ContrastiveLoss(nn.Module):
    """Contrastive loss for training on pairs"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, dist, label):
        """
        Args:
            dist: cosine distance between embeddings
            label: 1 for same person, 0 for different
        """
        # Same person: minimize distance
        # Different person: maximize distance (up to margin)
        loss_same = label * dist.pow(2)
        loss_diff = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return (loss_same + loss_diff).mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        emb1 = model.encode(img1)
        emb2 = model.encode(img2)
        
        # Calculate distance
        dist = cosine_distance(emb1, emb2)
        
        # Contrastive loss
        loss = criterion(dist, labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    distances, labels = [], []
    
    with torch.no_grad():
        for img1, img2, label in loader:
            emb1 = model.encode(img1.to(device))
            emb2 = model.encode(img2.to(device))
            dist = cosine_distance(emb1, emb2)
            distances.append(dist.cpu())
            labels.append(label)
    
    distances = torch.cat(distances).numpy()
    labels = torch.cat(labels).numpy()
    
    threshold, _ = find_best_threshold(distances, labels)
    preds = (distances <= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    
    return acc, threshold


def main():
    parser = argparse.ArgumentParser(description="Train using official LFW pairs")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--pairs-dir", default=None, type=str, help="Directory containing pairs files")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    info(f"Using device: {device}")

    # Get pairs files
    pairs_dir = Path(args.pairs_dir) if args.pairs_dir else ROOT
    try:
        train_pairs = get_lfw_pairs_path(pairs_dir, "train")
        test_pairs = get_lfw_pairs_path(pairs_dir, "test")
    except FileNotFoundError as e:
        info(f"Error: {e}")
        info(f"\nPlease download pairsDevTrain.txt and pairsDevTest.txt")
        info(f"from: http://vis-www.cs.umass.edu/lfw/")
        info(f"and place them in: {pairs_dir}")
        return

    processed_root = Path(cfg["paths"]["processed_root"])
    train_tfms, eval_tfms = build_transforms(cfg["data"]["image_size"], cfg["data"]["augment"])

    # Create dataloaders
    train_dataset = LFWPairsDataset(processed_root, train_pairs, transform=train_tfms)
    test_dataset = LFWPairsDataset(processed_root, test_pairs, transform=eval_tfms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    info(f"Training pairs: {len(train_dataset)}")
    info(f"Test pairs: {len(test_dataset)}")

    # Create model (no classifier needed for pairs training)
    model = FaceEncoder(
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        num_classes=None,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr_backbone"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    
    criterion = ContrastiveLoss(margin=cfg["train"]["triplet_margin"])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_acc = 0.0
    best_threshold = 0.5
    
    out_root = Path(cfg["paths"]["outputs"])
    checkpoints_dir = out_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    info("\nStarting training on official pairs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc, threshold = evaluate(model, test_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        info(f"Epoch {epoch}/{args.epochs} loss={train_loss:.4f} test_acc={test_acc:.4f} thr={threshold:.3f} lr={current_lr:.2e}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_threshold = threshold
            
            # Save checkpoint
            ckpt_path = checkpoints_dir / "best_pairs_trained.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "best_threshold": best_threshold,
                "epoch": epoch,
                "test_accuracy": test_acc,
            }, ckpt_path)
            info(f"  ✓ Saved best model: {ckpt_path}")
        
        scheduler.step()

    info("\n" + "="*60)
    info(f"Training complete!")
    info(f"Best test accuracy: {best_acc:.4f} at threshold {best_threshold:.3f}")
    info(f"Model saved to: {checkpoints_dir / 'best_pairs_trained.pt'}")
    info("="*60)


if __name__ == "__main__":
    main()
