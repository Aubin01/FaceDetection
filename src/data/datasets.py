import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: Path
    identity: str
    label: int | None  # int for train classification, None otherwise


def load_manifest(processed_root: Path) -> Dict:
    manifest_path = processed_root / "splits.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}, run data prep first.")
    with manifest_path.open() as f:
        return json.load(f)


def _collect_samples(processed_root: Path, identities: Sequence[str], label_map: Dict[str, int] | None) -> List[Sample]:
    samples: List[Sample] = []
    for identity in identities:
        identity_dir = processed_root / identity
        if not identity_dir.exists():
            continue
        for img_path in identity_dir.glob("*.png"):
            label = label_map.get(identity) if label_map else None
            samples.append(Sample(path=img_path, identity=identity, label=label))
    return samples


class AlignedFaceDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        transform: Callable,
    ):
        self.processed_root = Path(processed_root)
        self.transform = transform
        manifest = load_manifest(self.processed_root)

        if split == "train":
            identities = manifest["train_identities"]
            label_map = manifest["train_class_to_idx"]
        elif split == "val":
            identities = manifest["val_identities"]
            label_map = None
        elif split == "test":
            identities = manifest["test_identities"]
            label_map = None
        else:
            raise ValueError(f"Invalid split: {split}")

        self.samples = _collect_samples(self.processed_root, identities, label_map)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        return self.transform(img), sample.label, sample.identity


class PairDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        identities: Sequence[str],
        transform: Callable,
        max_pairs_per_identity: int | None = 5,
        seed: int = 42,
    ):
        self.processed_root = Path(processed_root)
        self.transform = transform
        self.seed = seed

        id_to_images = {iden: sorted((self.processed_root / iden).glob("*.png")) for iden in identities}
        self.pairs = self._build_pairs(id_to_images, max_pairs_per_identity)

    def _build_pairs(
        self, id_to_images: Dict[str, List[Path]], max_pairs_per_identity: int | None
    ) -> List[Tuple[Path, Path, int]]:
        random.seed(self.seed)
        pairs: List[Tuple[Path, Path, int]] = []
        identities = list(id_to_images.keys())

        for identity, images in id_to_images.items():
            if len(images) < 2:
                continue
            pos_combos = list(combinations(images, 2))
            if max_pairs_per_identity:
                random.shuffle(pos_combos)
                pos_combos = pos_combos[:max_pairs_per_identity]
            for img_a, img_p in pos_combos:
                pairs.append((img_a, img_p, 1))

                # negative pair using a different identity
                neg_id = random.choice([i for i in identities if i != identity])
                neg_img = random.choice(id_to_images[neg_id])
                pairs.append((img_a, neg_img, 0))
        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_b, label = self.pairs[idx]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        return self.transform(img_a), self.transform(img_b), label
