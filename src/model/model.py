# Description: Face embedding network plus training losses and distance helpers.
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceEncoder(nn.Module):
    """FaceNet-style encoder that outputs L2-normalized embeddings."""

    def __init__(self, embedding_dim: int, dropout: float = 0.5, num_classes: int | None = None, pretrained: bool = True):
        super().__init__()
        weights = "vggface2" if pretrained else None
        self.backbone = InceptionResnetV1(pretrained=weights, classify=False)
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor]:
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding) if self.classifier else None
        return logits, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, embedding = self.forward(x)
        return embedding


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets)


def triplet_random_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Build random triplets inside a batch to encourage separation."""
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels.cpu().tolist()):
        label_to_indices[label].append(idx)

    if len(label_to_indices) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    triplets = []
    labels_list = list(label_to_indices.keys())
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        for anchor_idx in indices:
            pos_candidates = [i for i in indices if i != anchor_idx]
            if not pos_candidates:
                continue
            positive_idx = random.choice(pos_candidates)
            neg_label = random.choice([l for l in labels_list if l != label])
            negative_idx = random.choice(label_to_indices[neg_label])
            triplets.append((anchor_idx, positive_idx, negative_idx))

    if not triplets:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    anc = torch.stack([embeddings[a] for a, _, _ in triplets])
    pos = torch.stack([embeddings[p] for _, p, _ in triplets])
    neg = torch.stack([embeddings[n] for _, _, n in triplets])

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    return loss_fn(anc, pos, neg)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance where 0 means identical and 2 means opposite."""
    return 1.0 - F.cosine_similarity(a, b)
