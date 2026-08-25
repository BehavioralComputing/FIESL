from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class EvidenceBatch:
    account_ids: list[str]
    labels: torch.Tensor
    typed_inputs: torch.Tensor
    input_dims: torch.Tensor
    availability_mask: torch.Tensor
    quality_features: torch.Tensor

    def validate(self, check_values: bool = True) -> "EvidenceBatch":
        if self.typed_inputs.ndim != 3 or self.typed_inputs.shape[1] != 9:
            raise ValueError("typed_inputs must have shape [N, 9, F]")
        count, slots, width = self.typed_inputs.shape
        if len(self.account_ids) != count or self.labels.shape != (count,):
            raise ValueError("account and label dimensions do not match")
        if self.input_dims.shape != (slots,):
            raise ValueError("input_dims must have shape [9]")
        if self.availability_mask.shape != (count, slots):
            raise ValueError("availability_mask must have shape [N, 9]")
        if self.quality_features.ndim != 3 or self.quality_features.shape[:2] != (count, slots):
            raise ValueError("quality_features must have shape [N, 9, Q]")
        if self.typed_inputs.dtype != torch.float32:
            raise TypeError("typed_inputs must be float32")
        if self.quality_features.dtype != torch.float32:
            raise TypeError("quality_features must be float32")
        if self.labels.dtype != torch.int64 or self.input_dims.dtype != torch.int64:
            raise TypeError("labels and input_dims must be int64")
        if self.availability_mask.dtype != torch.bool:
            raise TypeError("availability_mask must be bool")
        if torch.any(self.input_dims <= 0) or torch.any(self.input_dims > width):
            raise ValueError("input_dims contains an invalid width")
        if check_values:
            if not torch.isfinite(self.typed_inputs).all() or not torch.isfinite(self.quality_features).all():
                raise ValueError("input tensors contain non-finite values")
            if torch.any(self.typed_inputs[~self.availability_mask] != 0):
                raise ValueError("unavailable evidence inputs must be zero")
            if torch.any(self.quality_features[~self.availability_mask] != 0):
                raise ValueError("unavailable evidence quality must be zero")
            if set(self.labels.tolist()) - {0, 1}:
                raise ValueError("labels must be binary")
            if len(set(self.account_ids)) != count:
                raise ValueError("account IDs must be unique within a split")
            dimensions = torch.arange(width).unsqueeze(0) >= self.input_dims.unsqueeze(1)
            if torch.any(self.typed_inputs.masked_select(dimensions.unsqueeze(0)) != 0):
                raise ValueError("values outside declared slot widths must be zero")
        return self

    def select(self, indices: torch.Tensor | list[int]) -> "EvidenceBatch":
        index = torch.as_tensor(indices, dtype=torch.int64)
        rows = index.tolist()
        return EvidenceBatch(
            account_ids=[self.account_ids[row] for row in rows],
            labels=self.labels.index_select(0, index),
            typed_inputs=self.typed_inputs.index_select(0, index),
            input_dims=self.input_dims,
            availability_mask=self.availability_mask.index_select(0, index),
            quality_features=self.quality_features.index_select(0, index),
        )

    def to(self, device: torch.device | str) -> "EvidenceBatch":
        return EvidenceBatch(
            account_ids=self.account_ids,
            labels=self.labels.to(device),
            typed_inputs=self.typed_inputs.to(device),
            input_dims=self.input_dims.to(device),
            availability_mask=self.availability_mask.to(device),
            quality_features=self.quality_features.to(device),
        )


class EvidenceSplit(Dataset[int]):
    def __init__(self, batch: EvidenceBatch) -> None:
        self.batch = batch.validate(check_values=False)

    def __len__(self) -> int:
        return len(self.batch.account_ids)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def collate(self, indices: list[int]) -> EvidenceBatch:
        return self.batch.select(indices)


def load_split(path: Path, expected_input_dims: list[int], quality_dim: int) -> EvidenceBatch:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {
        "account_ids",
        "labels",
        "typed_inputs",
        "input_dims",
        "availability_mask",
        "quality_features",
    }
    if set(payload) < required:
        raise ValueError(f"{path} is missing keys: {sorted(required - set(payload))}")
    batch = EvidenceBatch(
        account_ids=[str(value) for value in payload["account_ids"]],
        labels=payload["labels"],
        typed_inputs=payload["typed_inputs"],
        input_dims=payload["input_dims"],
        availability_mask=payload["availability_mask"],
        quality_features=payload["quality_features"],
    ).validate(check_values=False)
    if batch.input_dims.tolist() != [int(value) for value in expected_input_dims]:
        raise ValueError(f"{path} input dimensions differ from the configuration")
    if batch.quality_features.shape[2] != int(quality_dim):
        raise ValueError(f"{path} quality dimension differs from the configuration")
    return batch


def make_loader(
    batch: EvidenceBatch,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[EvidenceBatch]:
    dataset = EvidenceSplit(batch)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
