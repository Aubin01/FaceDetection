import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.prepare import prepare_dataset
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Download, align, and split LFW faces.")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
