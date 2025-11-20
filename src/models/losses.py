import random
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets)


def triplet_random_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Builds random anchor/positive/negative triplets inside a batch."""
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
