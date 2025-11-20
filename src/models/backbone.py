import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FaceEncoder(nn.Module):
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

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding) if self.classifier else None
        return logits, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, embedding = self.forward(x)
        return embedding
