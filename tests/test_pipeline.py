from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as functional

from fiesl.data import load_split
from fiesl.prepare import prepare
from fiesl.training import run


class TestEncoder:
    dimension = 768

    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(list(digest), dtype=torch.float32).repeat(24)
            rows.append(functional.normalize(values - values.mean(), dim=0))
        return torch.stack(rows) if rows else torch.empty((0, 768), dtype=torch.float32)

    def manifest(self) -> dict[str, object]:
        return {"kind": "test_fixture", "hidden_size": 768}


def configuration(dataset: str) -> dict[str, object]:
    return {
        "method": "FIESL",
        "dataset": dataset,
        "expected_split_counts": {"train": 4, "dev": 2, "test": 2} if dataset == "TwiBot-20" else {"train": 2, "dev": 1, "test": 1},
        "representation_root": f"data/processed/{dataset}",
        "output_root": f"outputs/{dataset}",
        "seeds_file": "configs/seeds.json",
        "default_seed": 7,
        "input_dims": [783, 773, 1, 4, 4, 4, 768, 9, 15],
        "quality_dim": 8,
        "text_encoding": {
            "huggingface_model": "FacebookAI/roberta-base",
            "huggingface_url": "https://huggingface.co/FacebookAI/roberta-base",
            "resolved_revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            "local_model_path": None,
            "token_pooling": "masked_mean",
            "embedding_normalization": "l2",
            "tweet_pooling": "mean",
            "max_length": 128,
            "tweet_limit": None if dataset == "TwiBot-20" else 20,
            "scope": "all_own_tweets" if dataset == "TwiBot-20" else "first_20_own_tweets",
        },
        "model": {
            "hidden_dim": 8,
            "encoder_inner_dim": 16,
            "pair_hidden_dim": 8,
            "aggregation_hidden_dim": 16,
            "encoder_dropout": 0.0,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "evaluation_batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "minimum_epochs": 1,
            "patience": 1,
            "gradient_clip_norm": 5.0,
            "class_weighting": "inverse_frequency",
            "selection_split": "dev",
            "selection_metric": "dev_bot_f1",
            "classification_threshold": 0.5,
            "test_monitoring_per_epoch": True,
            "test_used_for_selection": False,
            "test_oracle_selection": False,
            "num_workers": 0,
            "device": "cpu",
        },
    }


def write_project(root: Path, dataset: str) -> Path:
    config_root = root / "configs"
    config_root.mkdir(parents=True)
    (config_root / "seeds.json").write_text(json.dumps({"seeds": [7]}), encoding="utf-8")
    path = config_root / "dataset.json"
    path.write_text(json.dumps(configuration(dataset)), encoding="utf-8")
    return path


def profile(index: int) -> dict[str, object]:
    return {
        "name": f"name {index}",
        "screen_name": f"screen_{index}",
        "protected": False,
        "verified": bool(index % 2),
        "has_extended_profile": True,
        "default_profile": False,
        "default_profile_image": False,
        "description": f"description {index}",
        "location": "Earth",
        "url": "https://example.org",
        "followers_count": 10 + index,
        "friends_count": 5 + index,
        "listed_count": index,
        "statuses_count": 100 + index,
        "favourites_count": 20 + index,
    }


def test_twibot20_from_source_to_training(tmp_path: Path) -> None:
    config_path = write_project(tmp_path, "TwiBot-20")
    raw = tmp_path / "raw"
    raw.mkdir()
    counts = {"train": 4, "dev": 2, "test": 2}
    account_index = 0
    for split, count in counts.items():
        rows = []
        for _ in range(count):
            rows.append({"ID": f"account-{account_index}", "label": str(account_index % 2), "profile": profile(account_index), "tweet": [f"tweet {account_index} a", f"tweet {account_index} b"]})
            account_index += 1
        (raw / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    output = prepare(config_path, raw, encoder=TestEncoder(), account_batch_size=2, text_batch_size=4)
    manifest = json.loads((output / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["contract"]["fit_split"] == "train"
    assert manifest["contract"]["tweet_selection"]["tweet_limit"] is None
    batch = load_split(output / "train.pt", list(configuration("TwiBot-20")["input_dims"]), 8)
    assert batch.typed_inputs.shape == (4, 9, 783)
    run_root = run(config_path)
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_completed"] == 1
    assert metrics["test_observation_count"] == 1


def write_csv(path: Path, field: str, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", field])
        writer.writeheader()
        for account_id, value in rows:
            writer.writerow({"id": account_id, field: value})


def test_twibot22_official_files_and_first_twenty(tmp_path: Path) -> None:
    config_path = write_project(tmp_path, "TwiBot-22")
    raw = tmp_path / "raw"
    raw.mkdir()
    split_rows = [("u1", "train"), ("u2", "train"), ("u3", "val"), ("u4", "test")]
    label_rows = [("u1", "human"), ("u2", "bot"), ("u3", "human"), ("u4", "bot")]
    write_csv(raw / "split.csv", "split", split_rows)
    write_csv(raw / "label.csv", "label", label_rows)
    users = []
    for index in range(1, 5):
        users.append({"id": f"u{index}", "name": f"name {index}", "username": f"user_{index}", "protected": False, "verified": False, "description": f"description {index}", "location": "Earth", "url": "https://example.org", "public_metrics": {"followers_count": index * 10, "following_count": index, "listed_count": index, "tweet_count": 30}})
    (raw / "user.json").write_text(json.dumps(users), encoding="utf-8")
    tweets = [{"id": f"t{index}", "author_id": "u1", "text": f"tweet {index}"} for index in range(22)]
    tweets.extend({"id": f"u{user}-t", "author_id": f"u{user}", "text": f"tweet user {user}"} for user in range(2, 5))
    (raw / "tweet_0.json").write_text(json.dumps(tweets), encoding="utf-8")
    for part in range(1, 9):
        (raw / f"tweet_{part}.json").write_text("[]", encoding="utf-8")
    (raw / "edge.csv").write_text("this file must never be parsed", encoding="utf-8")
    output = prepare(config_path, raw, encoder=TestEncoder(), account_batch_size=2, text_batch_size=8)
    connection = sqlite3.connect(output / "twibot22_preparation.sqlite")
    selected = connection.execute("SELECT COUNT(*) FROM tweets WHERE account_id = 'u1'").fetchone()[0]
    connection.close()
    assert selected == 20
    manifest = json.loads((output / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["tweet_selection"]["tweet_limit"] == 20
    assert manifest["contract"]["topology_files_opened"] == []
    assert manifest["contract"]["split_counts"] == {"train": 2, "dev": 1, "test": 1}
