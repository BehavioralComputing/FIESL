from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest
import torch

from fiesl.model import FIESL


ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = runpy.run_path(str(ROOT / "scripts/map_raw_fields_to_units.py"))


def test_frozen_compilation_matches_training_membership() -> None:
    response, provenance = NAMESPACE["load_frozen_compilation"]()
    compiled = NAMESPACE["compile_mapping"](response, provenance)
    model = FIESL([783, 773, 1, 4, 4, 4, 768, 9, 15], 8)
    expected = torch.tensor(compiled["training_membership"], dtype=torch.float32)
    assert compiled["audit"]["status"] == "PASS"
    assert torch.equal(model.unit_membership.cpu(), expected)


def test_compiler_rejects_missing_raw_field() -> None:
    response, _ = NAMESPACE["load_frozen_compilation"]()
    invalid = copy.deepcopy(response)
    invalid["field_alignments"]["TwiBot-20"].pop()
    with pytest.raises(ValueError, match="raw_field_coverage_exactly_once"):
        NAMESPACE["compile_mapping"](invalid)


def test_prompt_requests_both_mapping_levels() -> None:
    prompt = NAMESPACE["build_prompt"]()
    assert "field_alignments" in prompt
    assert "units" in prompt
    assert "Dataset raw-field metadata" in prompt
