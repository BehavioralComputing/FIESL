from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch

from fiesl.data import EvidenceBatch


def payload_batch(payload: dict[str, Any]) -> EvidenceBatch:
    return EvidenceBatch(
        account_ids=[str(value) for value in payload["account_ids"]],
        labels=payload["labels"].to(torch.int64),
        typed_inputs=payload["typed_inputs"].to(torch.float32),
        input_dims=payload["input_dims"].to(torch.int64),
        availability_mask=payload["availability_mask"].to(torch.bool),
        quality_features=payload["quality_features"].to(torch.float32),
    ).validate()


def concatenate(batches: list[EvidenceBatch]) -> EvidenceBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    first = batches[0]
    if any(not torch.equal(first.input_dims, batch.input_dims) for batch in batches[1:]):
        raise ValueError("shard input dimensions differ")
    return EvidenceBatch(
        account_ids=[value for batch in batches for value in batch.account_ids],
        labels=torch.cat([batch.labels for batch in batches]),
        typed_inputs=torch.cat([batch.typed_inputs for batch in batches]),
        input_dims=first.input_dims,
        availability_mask=torch.cat([batch.availability_mask for batch in batches]),
        quality_features=torch.cat([batch.quality_features for batch in batches]),
    ).validate()


def pin(batch: EvidenceBatch) -> EvidenceBatch:
    return EvidenceBatch(
        account_ids=batch.account_ids,
        labels=batch.labels.pin_memory(),
        typed_inputs=batch.typed_inputs.pin_memory(),
        input_dims=batch.input_dims.pin_memory(),
        availability_mask=batch.availability_mask.pin_memory(),
        quality_features=batch.quality_features.pin_memory(),
    ).validate()


class ShardedEvidenceBatches:
    def __init__(
        self,
        representation_root: Path,
        entries: list[dict[str, Any]],
        batch_size: int,
        seed: int,
        shuffle: bool,
        coalesce: bool = True,
        prefetch: bool = True,
        pin_memory: bool = False,
    ) -> None:
        if batch_size <= 0 or not entries:
            raise ValueError("sharded loader requires shards and a positive batch size")
        self.root = representation_root
        self.entries = list(entries)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.coalesce = bool(coalesce)
        self.prefetch = bool(prefetch)
        self.pin_memory = bool(pin_memory)
        self.iteration = 0

    def load(self, entry: dict[str, Any]) -> EvidenceBatch:
        path = self.root / str(entry["file"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        batch = payload_batch(payload)
        if len(batch.account_ids) != int(entry["records"]):
            raise ValueError(f"shard record count differs: {path}")
        return batch

    def loaded(self, entries: list[dict[str, Any]]) -> Iterator[EvidenceBatch]:
        if not self.prefetch:
            for entry in entries:
                yield self.load(entry)
            return
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.load, entries[0])
            for index in range(len(entries)):
                batch = future.result()
                if index + 1 < len(entries):
                    future = pool.submit(self.load, entries[index + 1])
                yield batch

    def __iter__(self) -> Iterator[EvidenceBatch]:
        rng = random.Random(self.seed + self.iteration if self.shuffle else self.seed)
        self.iteration += 1
        entries = list(self.entries)
        if self.shuffle:
            rng.shuffle(entries)
        pending: list[EvidenceBatch] = []
        pending_count = 0
        for batch in self.loaded(entries):
            indices = list(range(len(batch.account_ids)))
            if self.shuffle:
                rng.shuffle(indices)
            if not self.coalesce:
                for start in range(0, len(indices), self.batch_size):
                    value = batch.select(indices[start : start + self.batch_size])
                    yield pin(value) if self.pin_memory else value
                continue
            offset = 0
            while offset < len(indices):
                take = min(self.batch_size - pending_count, len(indices) - offset)
                pending.append(batch.select(indices[offset : offset + take]))
                pending_count += take
                offset += take
                if pending_count == self.batch_size:
                    value = concatenate(pending)
                    yield pin(value) if self.pin_memory else value
                    pending = []
                    pending_count = 0
        if pending:
            value = concatenate(pending)
            yield pin(value) if self.pin_memory else value
