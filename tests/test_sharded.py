from __future__ import annotations

from pathlib import Path

import torch

from fiesl.sharded import ShardedEvidenceBatches


def write_shard(root: Path, name: str, start: int, count: int) -> dict[str, object]:
    width = 783
    typed = torch.zeros((count, 9, width), dtype=torch.float32)
    typed[:, 0, 0] = torch.arange(start, start + count, dtype=torch.float32)
    payload = {
        "account_ids": [f"u{value}" for value in range(start, start + count)],
        "labels": torch.tensor([(start + value) % 2 for value in range(count)], dtype=torch.int64),
        "typed_inputs": typed,
        "input_dims": torch.tensor([783, 773, 1, 4, 4, 4, 768, 9, 15]),
        "availability_mask": torch.ones((count, 9), dtype=torch.bool),
        "quality_features": torch.zeros((count, 9, 8), dtype=torch.float32),
    }
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"file": name, "records": count}


def test_sharded_loader_coalesces_and_reiterates(tmp_path: Path) -> None:
    entries = [write_shard(tmp_path, "shards/a.pt", 0, 2), write_shard(tmp_path, "shards/b.pt", 2, 4)]
    loader = ShardedEvidenceBatches(tmp_path, entries, 3, 7, True, True, False, False)
    first = [value for batch in loader for value in batch.account_ids]
    second = [value for batch in loader for value in batch.account_ids]
    assert set(first) == {f"u{value}" for value in range(6)}
    assert set(second) == set(first)
    assert first != second
