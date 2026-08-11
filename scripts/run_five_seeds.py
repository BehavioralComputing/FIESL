from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds_path = project_root / config["seeds_file"]
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))["seeds"]
    for seed in seeds:
        command = [
            sys.executable,
            "-m",
            "fiesl.training",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
        ]
        if args.output_root is not None:
            command.extend(("--output-root", str(args.output_root.resolve())))
        subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()

