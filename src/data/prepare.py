import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.datasets import ImageFolder, LFWPeople
from tqdm import tqdm

from src.utils.config import ensure_dir
from src.utils.device import get_device
from src.utils.logging import info
from src.utils.seed import set_seed


def _download_lfw_torchvision(root: Path) -> Path:
    target = root / "lfw-deepfunneled"
    if target.exists():
        info(f"Found existing LFW data at {target}")
        return target

    info("Downloading LFW (deepfunneled) via torchvision...")
    _ = LFWPeople(root=root, split="train", image_set="deepfunneled", download=True)
    if not target.exists():
        raise RuntimeError("LFW download finished but lfw-deepfunneled not found.")
    return target


def _download_lfw_kagglehub(dataset: str, subdir: str | None) -> Path:
    info(f"Downloading LFW via kagglehub dataset_download('{dataset}')...")
    path = Path(kagglehub.dataset_download(dataset))
    info(f"Kagglehub dataset located at {path}")
    if subdir:
        candidate = path / subdir
        if not candidate.exists():
            raise FileNotFoundError(f"Kagglehub path {candidate} not found. Check kaggle_subdir in config.")
        return candidate

    # Heuristic search if no subdir specified
    for name in ["lfw-deepfunneled", "lfw_funneled", "lfw-deepfunneled.tgz", "lfw.tgz"]:
        matches = list(path.rglob(name))
        if matches:
            match = matches[0]
            if match.suffix in {".tgz", ".tar", ".gz"}:
                raise FileNotFoundError(
                    f"Found archive {match} in kagglehub dataset. Extract it and set kaggle_subdir to the extracted folder."
                )
            return match

    # Fallback: return root if it looks like ImageFolder
    if any(path.iterdir()):
        return path
    raise FileNotFoundError("Could not locate LFW images in kagglehub dataset. Set kaggle_subdir explicitly.")


def _align_dataset(
    lfw_root: Path,
    processed_root: Path,
    image_size: int,
    mtcnn_kwargs: Dict,
    device_str: str,
) -> Tuple[int, int]:
    processed_root.mkdir(parents=True, exist_ok=True)
    dataset = ImageFolder(lfw_root)
    device = get_device(device_str)
    mtcnn = MTCNN(image_size=image_size, device=device, **mtcnn_kwargs)

    saved, skipped = 0, 0
    info("Aligning faces with MTCNN (this may take a few minutes)...")
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

    info(f"Aligned faces saved: {saved}, skipped (no detection): {skipped}")
    return saved, skipped


def _build_splits(
    processed_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_images: int,
    seed: int,
) -> Dict:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "splits must sum to 1"
    identities = []
    for identity_dir in processed_root.iterdir():
        if not identity_dir.is_dir():
            continue
        images = list(identity_dir.glob("*.png"))
        if len(images) >= min_images:
            identities.append(identity_dir.name)

    set_seed(seed)
    random.shuffle(identities)

    n = len(identities)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = identities[:n_train]
    val_ids = identities[n_train : n_train + n_val]
    test_ids = identities[n_train + n_val :]

    manifest = {
        "train_identities": sorted(train_ids),
        "val_identities": sorted(val_ids),
        "test_identities": sorted(test_ids),
        "train_class_to_idx": {name: idx for idx, name in enumerate(sorted(train_ids))},
        "stats": {
            "num_identities_total": len(identities),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "num_test": len(test_ids),
            "min_images_per_identity": min_images,
        },
    }
    return manifest


def prepare_dataset(config: Dict) -> Path:
    paths = config["paths"]
    data_root = ensure_dir(paths["data_root"])
    processed_root = ensure_dir(paths["processed_root"])
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
        device_str=config.get("device", "auto"),
    )

    manifest = _build_splits(
        processed_root=Path(processed_root),
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
        min_images=config["data"]["min_images_per_identity"],
        seed=config["seed"],
    )
    manifest_path = Path(processed_root) / "splits.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    info(f"Saved split manifest to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Prepare LFW dataset: download, align, split.")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    prepare_dataset(cfg)
