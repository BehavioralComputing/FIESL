from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from fiesl.data import EvidenceBatch


class OfficialRoBERTaOverlay:
    def __init__(self, representation_root: Path) -> None:
        account_ids = json.loads((representation_root / "official_account_ids.json").read_text(encoding="utf-8"))
        reuse = json.loads((representation_root / "official_reuse_manifest.json").read_text(encoding="utf-8"))
        provenance = reuse.get("text_encoding_provenance", {})
        if len(account_ids) != 1_000_000 or len(set(account_ids)) != 1_000_000:
            raise ValueError("official account order is invalid")
        if reuse.get("status") != "PASS" or reuse.get("bge_used") is not False or provenance.get("status") != "PASS":
            raise ValueError("official RoBERTa manifest is not PASS")
        self.row_by_account = {str(value): index for index, value in enumerate(account_ids)}
        text_root = representation_root / "official_text"
        description_path = text_root / "descriptions.float32.mmap"
        tweets_path = text_root / "tweets.float32.mmap"
        counts_path = text_root / "tweet_counts.uint32.mmap"
        expected = 1_000_000 * 768
        if description_path.stat().st_size != expected * 4 or tweets_path.stat().st_size != expected * 4 or counts_path.stat().st_size != 4_000_000:
            raise ValueError("official RoBERTa tensor size differs")
        self.descriptions = np.memmap(description_path, dtype=np.float32, mode="r", shape=(1_000_000, 768))
        self.tweets = np.memmap(tweets_path, dtype=np.float32, mode="r", shape=(1_000_000, 768))
        self.counts = np.memmap(counts_path, dtype=np.uint32, mode="r", shape=(1_000_000,))
        self.contract = {
            "encoding_variant": "official_roberta_supported_fields",
            "profile_source": "official_description",
            "content_source": "official_first_20_tweet_mean",
            "disabled_slots": ["Identity", "Content Diversity"],
            "bge_used": False,
        }

    def apply(self, batch: EvidenceBatch) -> EvidenceBatch:
        rows = np.fromiter((self.row_by_account[value] for value in batch.account_ids), dtype=np.int64, count=len(batch.account_ids))
        descriptions = torch.from_numpy(np.asarray(self.descriptions[rows]).copy())
        tweets = torch.from_numpy(np.asarray(self.tweets[rows]).copy())
        counts = torch.from_numpy(np.asarray(self.counts[rows]).copy().astype(np.int64))
        typed = batch.typed_inputs.clone()
        typed[:, 0, :] = 0
        typed[:, 1, :768] = descriptions
        typed[:, 6, :768] = tweets
        typed[:, 7, :] = 0
        available = batch.availability_mask.clone()
        available[:, 0] = False
        available[:, 1] = descriptions.abs().sum(dim=1) > 0
        available[:, 6] = counts > 0
        available[:, 7] = False
        quality = batch.quality_features.clone()
        quality[:, 0, :] = 0
        quality[:, 7, :] = 0
        content = (counts > 0).to(torch.float32)
        quality[:, 6, :] = 0
        quality[:, 6, 0] = content
        quality[:, 6, 1] = torch.log1p(counts.to(torch.float32))
        quality[:, 6, 5] = 1 - content
        quality[:, 6, 6] = 1
        quality = quality * available.unsqueeze(-1).to(quality.dtype)
        return EvidenceBatch(batch.account_ids, batch.labels, typed, batch.input_dims, available, quality).validate()


class OfficialRoBERTaBatches:
    def __init__(self, base: Iterable[EvidenceBatch], overlay: OfficialRoBERTaOverlay) -> None:
        self.base = base
        self.overlay = overlay

    def __iter__(self) -> Iterator[EvidenceBatch]:
        for batch in self.base:
            yield self.overlay.apply(batch)
