from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from fiesl.model import FIESL
from fiesl.official_overlay import OfficialRoBERTaOverlay
from fiesl.sharded import ShardedEvidenceBatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = project_root / config["representation_root"]
    manifest = json.loads((root / "representation_manifest.json").read_text(encoding="utf-8"))
    overlay = OfficialRoBERTaOverlay(root)
    batches = {}
    for split in ("train", "dev", "test"):
        loader = ShardedEvidenceBatches(root, manifest["split_shards"][split][:1], 512, 0, False, False, False, False)
        batch = overlay.apply(next(iter(loader)))
        batches[split] = batch
        print(split, len(batch.account_ids), tuple(batch.typed_inputs.shape), batch.availability_mask.sum(dim=0).tolist())
    if args.backward:
        device = torch.device(args.device)
        model = FIESL(input_dims=config["input_dims"], quality_dim=config["quality_dim"], **config["model"]).to(device)
        batch = batches["train"].to(device)
        loss = nn.CrossEntropyLoss()(model(batch), batch.labels)
        loss.backward()
        print("backward", float(loss.detach()), "PASS")


if __name__ == "__main__":
    main()
