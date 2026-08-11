from __future__ import annotations

import argparse
import json
from pathlib import Path

from fiesl.data import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    representation_root = project_root / config["representation_root"]
    manifest_path = representation_root / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    if manifest.get("status") != "PASS":
        raise ValueError("preparation manifest is not PASS")
    if not contract.get("interaction_free") or contract.get("topology_files_opened") != []:
        raise ValueError("preparation manifest violates the interaction-free contract")
    if contract.get("fit_split") != "train":
        raise ValueError("preprocessing was not fitted on Train")
    seen: set[str] = set()
    for split in ("train", "dev", "test"):
        batch = load_split(
            representation_root / f"{split}.pt",
            config["input_dims"],
            config["quality_dim"],
        )
        overlap = seen.intersection(batch.account_ids)
        if overlap:
            raise ValueError(f"account IDs overlap with an earlier split: {split}")
        seen.update(batch.account_ids)
        if contract.get("split_counts", {}).get(split) != len(batch.account_ids):
            raise ValueError(f"manifest split count differs for {split}")
        print(split, len(batch.account_ids), tuple(batch.typed_inputs.shape))


if __name__ == "__main__":
    main()
