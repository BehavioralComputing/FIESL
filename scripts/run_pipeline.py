from __future__ import annotations

import argparse
import json
from pathlib import Path

from fiesl.prepare import prepare
from fiesl.training import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-path", type=Path)
    group.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--account-batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--hash-source-files", action="store_true")
    parser.add_argument("--five-seeds", action="store_true")
    args = parser.parse_args()
    prepare(
        args.config,
        args.raw_root,
        model_path=args.model_path,
        model_name=args.model_name,
        model_revision=args.model_revision,
        hf_cache_dir=args.hf_cache_dir,
        device=args.device,
        account_batch_size=args.account_batch_size,
        text_batch_size=args.text_batch_size,
        hash_source_files=args.hash_source_files,
    )
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.five_seeds:
        seeds_path = config_path.parent.parent / config["seeds_file"]
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))["seeds"]
    else:
        seeds = [config["default_seed"]]
    for seed in seeds:
        print(run(config_path, int(seed)))


if __name__ == "__main__":
    main()
