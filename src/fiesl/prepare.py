from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from fiesl.encoding import HuggingFaceTextEncoder, TextEncoder
from fiesl.features import INPUT_DIMS, QUALITY_ORDER, build_payload, fit_preprocessors
from fiesl.raw import Account, build_twibot22_index, iter_twibot22, read_twibot20


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_inventory(paths: Iterable[Path], hash_files: bool) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        item: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
        if hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 << 20):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        output.append(item)
    return output


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def merge_batches(batches: Iterable[list[Account]], count: int, encoder: TextEncoder, numeric: Any, style: Any, text_batch_size: int) -> dict[str, Any]:
    typed = torch.zeros((count, 9, 783), dtype=torch.float32)
    labels = torch.zeros(count, dtype=torch.int64)
    availability = torch.zeros((count, 9), dtype=torch.bool)
    quality = torch.zeros((count, 9, 8), dtype=torch.float32)
    account_ids: list[str] = []
    offset = 0
    for accounts in batches:
        payload = build_payload(accounts, encoder, numeric, style, text_batch_size)
        size = len(accounts)
        target = slice(offset, offset + size)
        typed[target] = payload["typed_inputs"]
        labels[target] = payload["labels"]
        availability[target] = payload["availability_mask"]
        quality[target] = payload["quality_features"]
        account_ids.extend(payload["account_ids"])
        offset += size
    if offset != count:
        raise ValueError(f"Representation row count mismatch: expected {count}, received {offset}")
    return {
        "account_ids": account_ids,
        "labels": labels,
        "typed_inputs": typed,
        "input_dims": torch.tensor(INPUT_DIMS, dtype=torch.int64),
        "availability_mask": availability,
        "quality_features": quality,
    }


def prepare(
    config_path: Path,
    raw_root: Path,
    *,
    model_path: Path | None = None,
    model_name: str | None = None,
    model_revision: str | None = None,
    hf_cache_dir: Path | None = None,
    device: str = "cuda",
    account_batch_size: int = 256,
    text_batch_size: int = 128,
    hash_source_files: bool = False,
    encoder: TextEncoder | None = None,
) -> Path:
    config_path = config_path.resolve()
    raw_root = raw_root.resolve()
    if account_batch_size <= 0 or account_batch_size > 900:
        raise ValueError("account_batch_size must be in [1, 900]")
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be positive")
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = str(config["dataset"])
    output_root = project_root / str(config["representation_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "preparation_manifest.json"
    if tuple(config["input_dims"]) != INPUT_DIMS or int(config["quality_dim"]) != len(QUALITY_ORDER):
        raise ValueError("Configuration differs from the public evidence contract")
    encoding = config["text_encoding"]
    model_revision = model_revision or encoding.get("resolved_revision")
    configured_local = encoding.get("local_model_path")
    if model_path is None and model_name is None and configured_local:
        candidate = Path(str(configured_local))
        model_path = candidate if candidate.is_absolute() else project_root / candidate
    source = model_path if model_path is not None else model_name or encoding["huggingface_model"]
    if dataset == "TwiBot-20" and (encoding.get("tweet_limit") is not None or encoding.get("scope") != "all_own_tweets"):
        raise ValueError("TwiBot-20 requires all own tweets")
    if dataset == "TwiBot-22" and (int(encoding.get("tweet_limit", 0)) != 20 or encoding.get("scope") != "first_20_own_tweets"):
        raise ValueError("TwiBot-22 requires the first 20 own tweets")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("contract", {})
        expected_selection = {"scope": encoding["scope"], "tweet_limit": encoding["tweet_limit"], "order": "official_dataset_order"}
        expected_counts = {name: int(value) for name, value in config["expected_split_counts"].items()}
        valid = (
            manifest.get("status") == "PASS"
            and manifest.get("contract_sha256") == canonical_hash(contract)
            and contract.get("dataset") == dataset
            and contract.get("max_length") == int(encoding["max_length"])
            and contract.get("tweet_selection") == expected_selection
            and contract.get("input_dims") == list(INPUT_DIMS)
            and contract.get("fit_split") == "train"
            and contract.get("interaction_free") is True
            and contract.get("topology_files_opened") == []
            and contract.get("split_counts") == expected_counts
            and all((output_root / f"{split}.pt").is_file() for split in ("train", "dev", "test"))
        )
        recorded_revision = contract.get("encoder", {}).get("resolved_revision")
        if recorded_revision is not None and recorded_revision != model_revision:
            valid = False
        if valid:
            return output_root
        raise RuntimeError(f"Existing preparation does not match the requested contract at {output_root}; move it aside before retrying")
    if encoder is None:
        encoder = HuggingFaceTextEncoder(
            source,
            max_length=int(encoding["max_length"]),
            device=device,
            cache_dir=hf_cache_dir,
            revision=model_revision,
            local_files_only=model_path is not None,
        )
    started = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, {"status": "RUNNING", "dataset": dataset, "started_at": started})
    if dataset == "TwiBot-20":
        source_paths = [raw_root / f"{split}.json" for split in ("train", "dev", "test")]
        accounts = read_twibot20(raw_root)
        numeric, style = fit_preprocessors(accounts["train"])
        counts = {split: len(rows) for split, rows in accounts.items()}
        for split in ("train", "dev", "test"):
            payload = merge_batches(
                (accounts[split][offset : offset + account_batch_size] for offset in range(0, len(accounts[split]), account_batch_size)),
                counts[split],
                encoder,
                numeric,
                style,
                text_batch_size,
            )
            save_payload(output_root / f"{split}.pt", payload)
            del payload
        selection = {"scope": "all_own_tweets", "tweet_limit": None, "order": "official_dataset_order"}
    elif dataset == "TwiBot-22":
        source_paths = [raw_root / "split.csv", raw_root / "label.csv", raw_root / "user.json"]
        source_paths.extend(raw_root / f"tweet_{index}.json" for index in range(9))
        index_path = output_root / "twibot22_preparation.sqlite"
        build_twibot22_index(raw_root, index_path, int(encoding["tweet_limit"]))
        numeric, style = fit_preprocessors(account for batch in iter_twibot22(index_path, "train", account_batch_size) for account in batch)
        import sqlite3

        connection = sqlite3.connect(index_path)
        counts = {split: int(connection.execute("SELECT COUNT(*) FROM accounts WHERE split = ?", (split,)).fetchone()[0]) for split in ("train", "dev", "test")}
        connection.close()
        for split in ("train", "dev", "test"):
            payload = merge_batches(iter_twibot22(index_path, split, account_batch_size), counts[split], encoder, numeric, style, text_batch_size)
            save_payload(output_root / f"{split}.pt", payload)
            del payload
        selection = {"scope": "first_20_own_tweets", "tweet_limit": 20, "order": "official_dataset_order"}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    expected_counts = {name: int(value) for name, value in config["expected_split_counts"].items()}
    if counts != expected_counts:
        raise ValueError(f"Official split counts differ: expected {expected_counts}, received {counts}")
    overlaps = set()
    split_ids = {}
    for split in ("train", "dev", "test"):
        payload = torch.load(output_root / f"{split}.pt", map_location="cpu", weights_only=False)
        ids = set(payload["account_ids"])
        if len(ids) != len(payload["account_ids"]):
            raise ValueError(f"Duplicate account IDs in {split}")
        overlaps |= ids.intersection(set().union(*split_ids.values())) if split_ids else set()
        split_ids[split] = ids
        del payload
    if overlaps:
        raise ValueError("Official splits overlap")
    contract = {
        "dataset": dataset,
        "encoder": encoder.manifest(),
        "token_pooling": "masked_mean",
        "embedding_normalization": "l2",
        "tweet_pooling": "mean",
        "max_length": int(encoding["max_length"]),
        "tweet_selection": selection,
        "input_dims": list(INPUT_DIMS),
        "quality_dimension": len(QUALITY_ORDER),
        "numeric_preprocessor": numeric.state_dict(),
        "linguistic_style_preprocessor": style.state_dict(),
        "fit_split": "train",
        "topology_files_opened": [],
        "interaction_free": True,
        "split_counts": counts,
        "source_files": file_inventory(source_paths, hash_source_files),
    }
    manifest = {
        "status": "PASS",
        "schema_name": "fiesl_public_preparation",
        "schema_version": 1,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
    }
    write_json(manifest_path, manifest)
    return output_root


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
    args = parser.parse_args()
    result = prepare(
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
    print(result)


if __name__ == "__main__":
    main()
