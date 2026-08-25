from __future__ import annotations

import io
import json
import runpy
import tokenize
from pathlib import Path

import torch

from fiesl.data import EvidenceBatch
from fiesl.model import FIESL


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SEEDS = [1503191042, 1284006632, 2049683099, 375449128, 545955441]
EXPECTED_DEFAULT_SEEDS = {
    "fiesl_twibot20.json": 375449128,
    "fiesl_twibot22.json": 545955441,
}
EXPECTED_SPLIT_COUNTS = {
    "fiesl_twibot20.json": {"train": 8278, "dev": 2365, "test": 1183},
    "fiesl_twibot22.json": {"train": 700000, "dev": 200000, "test": 100000},
}


def validate_names() -> None:
    forbidden = [
        "".join(("c", "1", "rr")),
        "".join(("c", "1", "-", "rr")),
        "".join(("chat", "gpt")),
        "".join(("generated", " by ai")),
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() not in {".py", ".json", ".md", ".toml"}:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        for value in forbidden:
            if value in lowered:
                raise ValueError(f"forbidden development term in {path.relative_to(ROOT)}")


def validate_comments() -> None:
    for path in ROOT.rglob("*.py"):
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
        if any(token.type == tokenize.COMMENT for token in tokens):
            raise ValueError(f"source comment found in {path.relative_to(ROOT)}")


def validate_configs() -> None:
    seeds = json.loads((ROOT / "configs/seeds.json").read_text(encoding="utf-8"))["seeds"]
    if seeds != EXPECTED_SEEDS:
        raise ValueError("seed configuration differs from the paper protocol")
    for name in ("fiesl_twibot20.json", "fiesl_twibot22.json"):
        config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        if config["method"] != "FIESL":
            raise ValueError("public method name is not FIESL")
        if config["default_seed"] != EXPECTED_DEFAULT_SEEDS[name]:
            raise ValueError("dataset default seed changed")
        if config["expected_split_counts"] != EXPECTED_SPLIT_COUNTS[name]:
            raise ValueError("official split counts changed")
        if config["default_seed"] not in seeds:
            raise ValueError("dataset default seed is outside the paper seed list")
        encoding = config["text_encoding"]
        if encoding["huggingface_model"] != "FacebookAI/roberta-base":
            raise ValueError("official encoder changed")
        if encoding["huggingface_url"] != "https://huggingface.co/FacebookAI/roberta-base":
            raise ValueError("official encoder URL changed")
        if encoding["resolved_revision"] != "e2da8e2f811d1448a5b465c236feacd80ffbac7b":
            raise ValueError("official encoder revision changed")
        if encoding["max_length"] != 128:
            raise ValueError("official maximum text length changed")
        training = config["training"]
        if training["selection_split"] != "dev" or training["selection_metric"] != "dev_bot_f1":
            raise ValueError("Dev Bot-F1 selection contract changed")
        if not training["test_monitoring_per_epoch"]:
            raise ValueError("per-epoch Test observation is disabled")
        if training["test_used_for_selection"] or training["test_oracle_selection"]:
            raise ValueError("Test selection is enabled")


def validate_pipeline_files() -> None:
    required = (
        "scripts/prepare_data.py",
        "scripts/run_pipeline.py",
        "scripts/map_raw_fields_to_units.py",
        "configs/frozen_schema_compilation.json",
        "src/fiesl/encoding.py",
        "src/fiesl/features.py",
        "src/fiesl/prepare.py",
        "src/fiesl/raw.py",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError(f"from-source pipeline files are missing: {missing}")


def validate_model() -> None:
    input_dims = [783, 773, 1, 4, 4, 4, 768, 9, 15]
    count = 3
    typed = torch.zeros((count, 9, 783), dtype=torch.float32)
    availability = torch.ones((count, 9), dtype=torch.bool)
    availability[:, 2] = False
    quality = torch.zeros((count, 9, 8), dtype=torch.float32)
    for slot, width in enumerate(input_dims):
        if slot != 2:
            typed[:, slot, :width] = torch.randn(count, width)
            quality[:, slot] = torch.randn(count, 8)
    batch = EvidenceBatch(
        account_ids=[str(index) for index in range(count)],
        labels=torch.tensor([0, 1, 0], dtype=torch.int64),
        typed_inputs=typed,
        input_dims=torch.tensor(input_dims, dtype=torch.int64),
        availability_mask=availability,
        quality_features=quality,
    ).validate()
    model = FIESL(input_dims, 8)
    logits = model(batch)
    if logits.shape != (count, 2) or not torch.isfinite(logits).all():
        raise ValueError("FIESL smoke test failed")
    if model.pair_left.numel() != 36 or model.incidence.sum().item() != 72:
        raise ValueError("internal relation contract changed")


def validate_frozen_schema_compilation() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/map_raw_fields_to_units.py"))
    response, provenance = namespace["load_frozen_compilation"]()
    compiled = namespace["compile_mapping"](response, provenance)
    if compiled["audit"]["status"] != "PASS":
        raise ValueError("frozen schema compilation audit failed")
    model = FIESL([783, 773, 1, 4, 4, 4, 768, 9, 15], 8)
    expected = torch.tensor(compiled["training_membership"], dtype=torch.float32)
    if not torch.equal(model.unit_membership.cpu(), expected):
        raise ValueError("frozen compiler output differs from the training membership")


def main() -> None:
    validate_names()
    validate_comments()
    validate_configs()
    validate_pipeline_files()
    validate_model()
    validate_frozen_schema_compilation()
    print("release validation passed")


if __name__ == "__main__":
    main()
