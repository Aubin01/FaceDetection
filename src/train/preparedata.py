# Description: Data preparation and datasets for LFW face verification.
import json
import random
import tarfile
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import kagglehub
import numpy as np
import torch
import yaml
from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, LFWPeople
from tqdm import tqdm


# Utility functions
def get_device() -> torch.device:
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Manifest helpers
def load_manifest(processed_root: Path) -> Dict:
    """Load split manifest produced during data preparation."""
    manifest_path = processed_root / "splits.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}, run data prep first.")
    with manifest_path.open() as handle:
        return json.load(handle)


# Transforms
def build_transforms(image_size: int, augment_cfg: dict):
    """Training and eval transforms with light augmentation."""
    train_transforms = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment_cfg.get("horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get("rotation", 0):
        train_transforms.append(transforms.RandomRotation(augment_cfg["rotation"]))
    if augment_cfg.get("brightness") or augment_cfg.get("contrast"):
        brightness = augment_cfg.get("brightness", 0)
        contrast = augment_cfg.get("contrast", 0)
        train_transforms.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
    if augment_cfg.get("color_jitter", None):
        cj = augment_cfg["color_jitter"]
        train_transforms.append(transforms.ColorJitter(*cj))
    if augment_cfg.get("gaussian_blur"):
        train_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, augment_cfg["gaussian_blur"])))

    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return transforms.Compose(train_transforms), eval_transforms


#Datasets
def _collect_samples(processed_root: Path, identities: Sequence[str], label_map: Dict[str, int] | None) -> List[Tuple[Path, str, int | None]]:
    """Returns list of tuples: (path, identity, label)"""
    samples: List[Tuple[Path, str, int | None]] = []
    for identity in identities:
        identity_dir = processed_root / identity
        if not identity_dir.exists():
            continue
        for img_path in identity_dir.glob("*.png"):
            label = label_map.get(identity) if label_map else None
            samples.append((img_path, identity, label))
    return samples


class AlignedFaceDataset(Dataset):
    """Aligned images for classification pretraining."""

    def __init__(self, processed_root: str | Path, split: str, transform: Callable):
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
        path, identity, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, identity


class LFWPairsDataset(Dataset):
    """Official LFW pairs for verification using standard benchmark format."""

    def __init__(
        self,
        processed_root: str | Path,
        pairs_file: str | Path,
        transform: Callable,
    ):
        self.processed_root = Path(processed_root)
        self.transform = transform
        self.pairs = self._load_pairs(Path(pairs_file))

    def _load_pairs(self, pairs_file: Path) -> List[Tuple[Path, Path, int]]:
        """Load pairs from official LFW pairs.txt format."""
        pairs = []
        
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # First line contains number of pairs (skip it)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                # Positive pair: name img1 img2
                name = parts[0]
                img1_idx = int(parts[1])
                img2_idx = int(parts[2])
                
                # Format: name/name_0001.png (LFW uses 4-digit padding)
                path1 = self.processed_root / name / f"{name}_{img1_idx:04d}.png"
                path2 = self.processed_root / name / f"{name}_{img2_idx:04d}.png"
                
                if path1.exists() and path2.exists():
                    pairs.append((path1, path2, 1))
                else:
                    print(f"Warning: Skipping pair - files not found: {path1} or {path2}")
                    
            elif len(parts) == 4:
                # Negative pair: name1 img1 name2 img2
                name1 = parts[0]
                img1_idx = int(parts[1])
                name2 = parts[2]
                img2_idx = int(parts[3])
                
                path1 = self.processed_root / name1 / f"{name1}_{img1_idx:04d}.png"
                path2 = self.processed_root / name2 / f"{name2}_{img2_idx:04d}.png"
                
                if path1.exists() and path2.exists():
                    pairs.append((path1, path2, 0))
                else:
                    print(f"Warning: Skipping pair - files not found: {path1} or {path2}")
        
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_b, label = self.pairs[idx]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        return self.transform(img_a), self.transform(img_b), label


# Downloads and alignment
def _download_lfw_torchvision(root: Path) -> Path:
    target = root / "lfw-deepfunneled"
    if target.exists():
        print(f"Found existing LFW data at {target}")
        return target

    print("Downloading LFW (deepfunneled) via torchvision...")
    _ = LFWPeople(root=root, split="train", image_set="deepfunneled", download=True)
    if not target.exists():
        raise RuntimeError("LFW download finished but lfw-deepfunneled not found.")
    return target


def _download_lfw_kagglehub(dataset: str, subdir: str | None, data_root: Path) -> Path:
    print(f"Downloading LFW via kagglehub dataset_download('{dataset}')...")
    path = Path(kagglehub.dataset_download(dataset))
    print(f"Kagglehub dataset located at {path}")

    extract_dir = data_root / "lfw-funneled"

    if subdir:
        candidate = path / subdir
        if not candidate.exists():
            raise FileNotFoundError(f"Kagglehub path {candidate} not found. Check kaggle_subdir in config.")
        return candidate

    for name in ["lfw-funneled.tgz", "lfw-deepfunneled.tgz", "lfw.tgz"]:
        matches = list(path.rglob(name))
        if matches:
            archive = matches[0]
            print(f"Found archive {archive}, extracting to {extract_dir}...")
            if extract_dir.exists() and any(extract_dir.iterdir()):
                print(f"Already extracted at {extract_dir}")
                return extract_dir
            extract_dir.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=extract_dir.parent)
            if extract_dir.exists():
                return extract_dir
            for extracted in extract_dir.parent.iterdir():
                if extracted.is_dir() and "lfw" in extracted.name.lower():
                    print(f"Found extracted dataset at {extracted}")
                    return extracted
            raise FileNotFoundError(f"Extracted archive but couldn't find LFW folder in {extract_dir.parent}")

    for name in ["lfw-deepfunneled", "lfw_funneled", "lfw-funneled"]:
        matches = list(path.rglob(name))
        if matches:
            return matches[0]

    if any(path.iterdir()):
        return path
    raise FileNotFoundError("Could not locate LFW images in kagglehub dataset. Set kaggle_subdir explicitly.")


def _align_dataset(
    lfw_root: Path,
    processed_root: Path,
    image_size: int,
    mtcnn_kwargs: Dict,
) -> Tuple[int, int]:
    processed_root.mkdir(parents=True, exist_ok=True)
    dataset = ImageFolder(lfw_root)
    device = get_device()
    mtcnn = MTCNN(image_size=image_size, device=device, **mtcnn_kwargs)

    saved, skipped = 0, 0
    print("Aligning faces with MTCNN (this may take a few minutes)...")
    for path, target in tqdm(dataset.samples, desc="aligning", total=len(dataset)):
        identity = dataset.classes[target]
        dest_dir = processed_root / identity
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / (Path(path).stem + ".png")
        if dest_path.exists():
            saved += 1
            continue

        img = Image.open(path).convert("RGB")
        aligned = mtcnn(img, save_path=str(dest_path))
        if aligned is None:
            skipped += 1
            if dest_path.exists():
                dest_path.unlink()
            continue
        saved += 1

    print(f"Aligned faces saved: {saved}, skipped (no detection): {skipped}")
    return saved, skipped


def _build_train_manifest(processed_root: Path, min_images: int) -> Dict:
    """Build manifest with all identities for training classification."""
    identities = []
    for identity_dir in processed_root.iterdir():
        if not identity_dir.is_dir():
            continue
        images = list(identity_dir.glob("*.png"))
        if len(images) >= min_images:
            identities.append(identity_dir.name)

    train_ids = sorted(identities)
    manifest = {
        "train_identities": train_ids,
        "train_class_to_idx": {name: idx for idx, name in enumerate(train_ids)},
        "stats": {
            "num_identities_total": len(identities),
            "num_train": len(train_ids),
            "min_images_per_identity": min_images,
        },
    }
    return manifest


# Public entrypoint
def prepare_dataset(config: Dict) -> Path:
    """Download LFW, align with MTCNN, and save identity splits."""
    paths = config["paths"]
    data_root = Path(paths["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    processed_root = Path(paths["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)
    image_size = config["data"]["image_size"]

    mtcnn_kwargs = {
        "margin": config["mtcnn"].get("margin", 0),
        "thresholds": config["mtcnn"].get("thresholds", [0.6, 0.7, 0.7]),
        "post_process": config["mtcnn"].get("post_process", True),
    }

    data_cfg = config["data"]
    source = data_cfg.get("source", "kagglehub")
    if source == "kagglehub":
        lfw_root = _download_lfw_kagglehub(
            dataset=data_cfg.get("kaggle_dataset", "atulanandjha/lfwpeople"),
            subdir=data_cfg.get("kaggle_subdir"),
            data_root=Path(data_root),
        )
    elif source == "torchvision":
        lfw_root = _download_lfw_torchvision(Path(data_root))
    else:
        raise ValueError(f"Unsupported data.source '{source}', use 'kagglehub' or 'torchvision'.")

    _align_dataset(
        lfw_root=lfw_root,
        processed_root=Path(processed_root),
        image_size=image_size,
        mtcnn_kwargs=mtcnn_kwargs,
    )

    manifest = _build_train_manifest(
        processed_root=Path(processed_root),
        min_images=config["data"]["min_images_per_identity"],
    )
    manifest_path = Path(processed_root) / "splits.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved training manifest to {manifest_path}")
    print(f"Note: Validation/test use official LFW pairs from {config['paths'].get('split_root', 'split')}")
    return manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LFW dataset: download, align, split.")
    parser.add_argument("--config", default="src/model/config.yaml", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_data = yaml.safe_load(f)
    prepare_dataset(cfg_data)
