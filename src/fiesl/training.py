from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from fiesl.data import EvidenceBatch, load_split, make_loader
from fiesl.metrics import binary_metrics
from fiesl.model import FIESL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(
    model: nn.Module,
    loader: Iterable[EvidenceBatch],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    losses = 0.0
    count = 0
    labels = []
    probabilities = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.labels)
            size = len(batch.account_ids)
            losses += float(loss) * size
            count += size
            labels.append(batch.labels.cpu())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    result = binary_metrics(torch.cat(labels), torch.cat(probabilities), threshold)
    result["loss"] = losses / count
    return result


def train_epoch(
    model: nn.Module,
    loader: Iterable[EvidenceBatch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    losses = 0.0
    count = 0
    labels = []
    probabilities = []
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch.labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        size = len(batch.account_ids)
        losses += float(loss.detach()) * size
        count += size
        labels.append(batch.labels.detach().cpu())
        probabilities.append(torch.softmax(logits.detach(), dim=1)[:, 1].cpu())
    result = binary_metrics(torch.cat(labels), torch.cat(probabilities), threshold)
    result["loss"] = losses / count
    return result


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def run(config_path: Path, seed: int | None = None, output_override: Path | None = None) -> Path:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "FIESL":
        raise ValueError("the release runner supports FIESL only")
    training = config["training"]
    if training["selection_split"] != "dev" or training["selection_metric"] != "dev_bot_f1":
        raise ValueError("the release runner requires Dev Bot-F1 selection")
    seeds_path = resolve_path(project_root, config["seeds_file"])
    allowed_seeds = json.loads(seeds_path.read_text(encoding="utf-8"))["seeds"]
    seed = config["default_seed"] if seed is None else seed
    if seed not in allowed_seeds:
        raise ValueError("seed is not in the fixed paper seed list")
    set_seed(seed)
    representation_root = resolve_path(project_root, config["representation_root"])
    preparation_path = representation_root / "preparation_manifest.json"
    if not preparation_path.is_file():
        raise FileNotFoundError(preparation_path)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("status") != "PASS" or not preparation.get("contract_sha256"):
        raise ValueError("representation preparation is not a traceable PASS artifact")
    output_root = output_override or resolve_path(project_root, config["output_root"])
    run_id = f"FIESL_{config['dataset'].replace('-', '')}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "run_id": run_id,
        "method": "FIESL",
        "dataset": config["dataset"],
        "seed": seed,
        "config_sha256": config_hash,
        "preparation_contract_sha256": preparation["contract_sha256"],
        "selection_split": "dev",
        "selection_metric": "dev_bot_f1",
        "test_monitoring_per_epoch": True,
        "test_used_for_selection": False,
        "test_oracle_selection": False,
        "started_at": utc_now(),
    }
    write_json(run_root / "run_manifest.json", manifest)
    write_json(run_root / "status.json", {"status": "RUNNING", "run_id": run_id, "updated_at": utc_now()})
    input_dims = config["input_dims"]
    quality_dim = config["quality_dim"]
    train_batch = load_split(representation_root / "train.pt", input_dims, quality_dim)
    dev_batch = load_split(representation_root / "dev.pt", input_dims, quality_dim)
    test_batch = load_split(representation_root / "test.pt", input_dims, quality_dim)
    train_loader = make_loader(
        train_batch,
        training["batch_size"],
        True,
        seed,
        training["num_workers"],
    )
    dev_loader = make_loader(
        dev_batch,
        training["evaluation_batch_size"],
        False,
        seed,
        training["num_workers"],
    )
    test_loader = make_loader(
        test_batch,
        training["evaluation_batch_size"],
        False,
        seed,
        training["num_workers"],
    )
    requested_device = training["device"]
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    model = FIESL(input_dims=input_dims, quality_dim=quality_dim, **config["model"]).to(device)
    counts = torch.bincount(train_batch.labels, minlength=2).to(torch.float32)
    class_weights = train_batch.labels.numel() / (2 * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    threshold = training["classification_threshold"]
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = run_root / "checkpoint_best_dev.pt"
    for epoch in range(1, training["max_epochs"] + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            threshold,
            training["gradient_clip_norm"],
        )
        dev_metrics = evaluate(model, dev_loader, criterion, device, threshold)
        test_metrics = evaluate(model, test_loader, criterion, device, threshold)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "test_observation": test_metrics,
            "test_used_for_selection": False,
        }
        history.append(row)
        write_history(run_root / "history.jsonl", history)
        value = dev_metrics["f1"]
        if value > best_value:
            best_value = value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "config_sha256": config_hash,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if epoch >= training["minimum_epochs"] and stale_epochs >= training["patience"]:
            break
    if len(history) == 0 or best_epoch == 0:
        raise RuntimeError("training produced no completed epoch")
    selected = history[best_epoch - 1]
    metrics = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_dev_bot_f1": best_value,
        "dev_selected_test": selected["test_observation"],
        "test_observation_count": len(history),
        "test_used_for_selection": False,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(
        run_root / "status.json",
        {"status": "COMPLETED", "run_id": run_id, "updated_at": utc_now(), "epochs_completed": len(history)},
    )
    manifest["finished_at"] = utc_now()
    manifest["epochs_completed"] = len(history)
    manifest["best_epoch"] = best_epoch
    write_json(run_root / "run_manifest.json", manifest)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(run(args.config, args.seed, args.output_root))


if __name__ == "__main__":
    main()
