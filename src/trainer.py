from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.data.datasets import AlignedFaceDataset, PairDataset, load_manifest
from src.data.transforms import build_transforms
from src.models.backbone import FaceEncoder
from src.models.losses import cross_entropy_loss, triplet_random_loss
from src.utils.config import ensure_dir
from src.utils.device import get_device
from src.utils.logging import info
from src.utils.seed import set_seed


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b)


def find_best_threshold(distances: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.3, 1.2, 50)
    best_acc, best_thr = 0.0, 0.7
    for thr in thresholds:
        preds = (distances <= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


class FaceTrainer:
    def __init__(self, config: Dict):
        self.config = config
        set_seed(config["seed"])
        self.device = get_device(config.get("device", "auto"))
        info(f"Using device: {self.device}")

        processed_root = Path(config["paths"]["processed_root"])
        manifest = load_manifest(processed_root)
        self.manifest = manifest
        self.num_classes = len(manifest["train_identities"])

        train_tfms, eval_tfms = build_transforms(config["data"]["image_size"], config["data"]["augment"])
        batch_size = config["train"]["batch_size"]
        num_workers = config["train"]["num_workers"]

        self.train_loader = DataLoader(
            AlignedFaceDataset(processed_root, split="train", transform=train_tfms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_pairs = DataLoader(
            PairDataset(
                processed_root,
                identities=manifest["val_identities"],
                transform=eval_tfms,
                max_pairs_per_identity=5,
                seed=config["seed"],
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_pairs = DataLoader(
            PairDataset(
                processed_root,
                identities=manifest["test_identities"],
                transform=eval_tfms,
                max_pairs_per_identity=10,
                seed=config["seed"] + 1,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.model = FaceEncoder(
            embedding_dim=config["model"]["embedding_dim"],
            dropout=config["model"]["dropout"],
            num_classes=self.num_classes,
            pretrained=config["model"]["pretrained"],
        ).to(self.device)

        params = [
            {"params": list(self.model.backbone.parameters()), "lr": config["train"]["lr_backbone"]},
            {
                "params": list(self.model.embedding_head.parameters())
                + (list(self.model.classifier.parameters()) if self.model.classifier else []),
                "lr": config["train"]["lr_head"],
            },
        ]
        self.optimizer = torch.optim.Adam(params, weight_decay=config["train"]["weight_decay"])
        self.scheduler = None

        out_root = Path(config["paths"]["outputs"])
        self.checkpoints_dir = ensure_dir(out_root / "checkpoints")
        self.best_val_acc = 0.0
        self.best_threshold = 0.7

    def train(self):
        info("Stage 1: classification pretraining")
        self._train_classification()
        info("Stage 2: triplet fine-tuning")
        self._prepare_stage2_optimizer()
        self._train_triplet()
        info("Final evaluation on test pairs")
        test_acc, _ = self.evaluate(self.test_pairs, threshold=self.best_threshold)
        info(f"Test verification accuracy: {test_acc:.4f}")

    def _train_classification(self):
        epochs = self.config["train"]["epochs_classification"]
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for images, labels, _ in self.train_loader:
                labels = labels.to(self.device)
                images = images.to(self.device)

                self.optimizer.zero_grad()
                logits, embeddings = self.model(images)
                ce_loss = cross_entropy_loss(logits, labels)
                loss = ce_loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(self.train_loader))
            val_acc, thr = self.evaluate(self.val_pairs, threshold=None)
            info(f"[Classification] Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} thr={thr:.3f}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_threshold = thr
                self._save_checkpoint("best_classification.pt")

    def _prepare_stage2_optimizer(self):
        """Update optimizer with stage 2 learning rates and add scheduler"""
        config = self.config
        lr_backbone = config["train"].get("lr_backbone_stage2", config["train"]["lr_backbone"] * 0.5)
        lr_head = config["train"].get("lr_head_stage2", config["train"]["lr_head"] * 0.5)
        
        params = [
            {"params": list(self.model.backbone.parameters()), "lr": lr_backbone},
            {
                "params": list(self.model.embedding_head.parameters())
                + (list(self.model.classifier.parameters()) if self.model.classifier else []),
                "lr": lr_head,
            },
        ]
        self.optimizer = torch.optim.Adam(params, weight_decay=config["train"]["weight_decay"])
        
        # Add cosine annealing scheduler for triplet training
        epochs_triplet = config["train"]["epochs_triplet"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs_triplet, eta_min=1e-6
        )
        info(f"Stage 2 optimizer: lr_backbone={lr_backbone:.2e}, lr_head={lr_head:.2e}")

    def _train_triplet(self):
        epochs = self.config["train"]["epochs_triplet"]
        margin = self.config["train"]["triplet_margin"]
        ce_weight = self.config["train"].get("triplet_ce_weight", 0.1)
        
        # Reset best for triplet stage
        best_triplet_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for images, labels, _ in self.train_loader:
                labels = labels.to(self.device)
                images = images.to(self.device)

                self.optimizer.zero_grad()
                logits, embeddings = self.model(images)
                ce_loss = cross_entropy_loss(logits, labels) * ce_weight
                triplet_loss = triplet_random_loss(embeddings, labels, margin=margin)
                loss = ce_loss + triplet_loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(self.train_loader))
            val_acc, thr = self.evaluate(self.val_pairs, threshold=None)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                info(f"[Triplet] Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} thr={thr:.3f} lr={current_lr:.2e}")
            else:
                info(f"[Triplet] Epoch {epoch}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f} thr={thr:.3f}")
            
            # Save best triplet model
            if val_acc > best_triplet_acc:
                best_triplet_acc = val_acc
                self.best_threshold = thr
                self._save_checkpoint("best_metric.pt")
            
            # Update overall best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

    def evaluate(self, loader: DataLoader, threshold: float | None = None) -> Tuple[float, float]:
        self.model.eval()
        distances, labels = [], []
        with torch.no_grad():
            for img_a, img_b, label in loader:
                emb_a = self.model.encode(img_a.to(self.device))
                emb_b = self.model.encode(img_b.to(self.device))
                dist = cosine_distance(emb_a, emb_b)
                distances.append(dist.cpu())
                labels.append(label)

        distances = torch.cat(distances).numpy()
        labels = torch.cat(labels).numpy()
        if threshold is None:
            thr, val_acc = find_best_threshold(distances, labels)
        else:
            thr = threshold
            preds = (distances <= thr).astype(int)
            val_acc = accuracy_score(labels, preds)
        return val_acc, thr

    def _save_checkpoint(self, name: str):
        ckpt_path = Path(self.checkpoints_dir) / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "best_threshold": self.best_threshold,
            },
            ckpt_path,
        )
        info(f"Saved checkpoint: {ckpt_path}")
