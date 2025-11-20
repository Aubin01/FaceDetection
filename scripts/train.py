import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.trainer import FaceTrainer


def main():
    parser = argparse.ArgumentParser(description="Train face recognition model (classification + triplet).")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = FaceTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
