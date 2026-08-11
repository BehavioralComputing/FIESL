from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as functional


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor: ...

    def manifest(self) -> dict[str, object]: ...


class HuggingFaceTextEncoder:
    def __init__(
        self,
        model_source: str | Path,
        *,
        max_length: int = 128,
        device: str = "cuda",
        cache_dir: Path | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the project dependencies before encoding") from error
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.source = str(model_source)
        self.max_length = int(max_length)
        self.device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
        options = {
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.source, **options)
        self.model = AutoModel.from_pretrained(self.source, **options).to(self.device).eval()
        self.dimension = int(self.model.config.hidden_size)
        if self.dimension != 768:
            raise ValueError("The official FIESL configuration requires a 768-dimensional encoder")
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return torch.empty((0, self.dimension), dtype=torch.float32)
        output = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                vectors = functional.normalize(pooled, p=2, dim=1, eps=1e-12)
            output.append(vectors.to(dtype=torch.float32, device="cpu"))
        return torch.cat(output, dim=0)

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "huggingface_transformer",
            "source": self.source,
            "resolved_revision": self.resolved_revision,
            "hidden_size": self.dimension,
            "max_length": self.max_length,
            "token_pooling": "masked_mean",
            "embedding_normalization": "l2",
            "trust_remote_code": False,
        }
