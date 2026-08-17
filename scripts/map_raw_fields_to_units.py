from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping


PRIMITIVE_ORDER = (
    "identity_lexical",
    "identity_metadata",
    "profile_text",
    "profile_completeness",
    "popularity",
    "social_ratio",
    "activity_intensity",
    "content_semantics",
    "content_diversity",
    "linguistic_style",
)

DATASET_PRIMITIVE_FIELDS = {
    "TwiBot-20": {
        "identity_lexical": ("profile.name", "profile.screen_name"),
        "identity_metadata": ("profile.protected", "profile.verified", "profile.has_extended_profile", "profile.default_profile", "profile.default_profile_image"),
        "profile_text": ("profile.description", "profile.location", "profile.url"),
        "profile_completeness": ("profile.description", "profile.location", "profile.url"),
        "popularity": ("profile.followers_count", "profile.listed_count"),
        "social_ratio": ("profile.followers_count", "profile.friends_count"),
        "activity_intensity": ("profile.statuses_count", "profile.favourites_count"),
        "content_semantics": ("tweet[]",),
        "content_diversity": ("tweet[]",),
        "linguistic_style": ("tweet[]",),
    },
    "TwiBot-22": {
        "identity_lexical": ("user.name", "user.username"),
        "identity_metadata": ("user.protected", "user.verified"),
        "profile_text": ("user.description", "user.location", "user.url"),
        "profile_completeness": ("user.description", "user.location", "user.url"),
        "popularity": ("user.public_metrics.followers_count", "user.public_metrics.listed_count"),
        "social_ratio": ("user.public_metrics.followers_count", "user.public_metrics.following_count"),
        "activity_intensity": ("user.public_metrics.tweet_count",),
        "content_semantics": ("tweet_*.json.text",),
        "content_diversity": ("tweet_*.json.text",),
        "linguistic_style": ("tweet_*.json.text",),
    },
}

FROZEN_UNITS = (
    {"id": "identity_lexical", "name": "Identity Lexical", "definition": "Lexical and character-form identity presentation.", "primitives": ("identity_lexical",)},
    {"id": "profile_text", "name": "Profile Text", "definition": "Free-text profile self-description and its length form.", "primitives": ("profile_text",)},
    {"id": "presentation_metadata", "name": "Presentation Metadata", "definition": "Identity flags and profile completeness indicators.", "primitives": ("identity_metadata", "profile_completeness")},
    {"id": "popularity", "name": "Popularity", "definition": "Follower and list-based social standing.", "primitives": ("popularity",)},
    {"id": "social_ratio", "name": "Social Ratio", "definition": "Follower and friend balance distinct from absolute standing.", "primitives": ("social_ratio",)},
    {"id": "activity", "name": "Activity", "definition": "Account-local activity intensity.", "primitives": ("activity_intensity",)},
    {"id": "content_semantics", "name": "Content Semantics", "definition": "Semantic summary of own tweets.", "primitives": ("content_semantics",)},
    {"id": "content_diversity", "name": "Content Diversity", "definition": "Dispersion and repetition of own tweets.", "primitives": ("content_diversity",)},
    {"id": "linguistic_style", "name": "Linguistic Style", "definition": "Label-free linguistic form of own tweets.", "primitives": ("linguistic_style",)},
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt() -> str:
    schemas = json.dumps(DATASET_PRIMITIVE_FIELDS, ensure_ascii=False, indent=2, sort_keys=True)
    primitives = ", ".join(PRIMITIVE_ORDER)
    return f"""You are a dataset-level semantic evidence schema agent. Map the normalized raw-field primitives shared by TwiBot-20 and TwiBot-22 into exactly nine semantic evidence units.

Allowed primitives:
{primitives}

Requirements:
- Return one JSON object with a units array.
- Every unit must contain id, name, definition, and primitives.
- Cover every allowed primitive exactly once across exactly nine units.
- Use only field-name metadata. Never inspect account values or account identifiers.
- Never use labels, Train/Dev/Test metrics, model results, graph edges, neighbors, timestamps, domain annotations, or relation-derived features.
- Do not propose encoders, transforms, edges, hyperparameters, or per-account decisions.
- Preserve content semantics, content diversity, and linguistic style as separate views of the same own-tweet source.

Dataset field metadata:
{schemas}
"""


def parse_response(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        text = value.strip()
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("response contains no JSON object")
            text = text[start : end + 1]
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def validate_units(value: str | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate = parse_response(value)
    units = candidate.get("units")
    valid_array = isinstance(units, (list, tuple)) and all(isinstance(unit, Mapping) for unit in units)
    units = list(units) if valid_array else []
    memberships = [primitive for unit in units for primitive in unit.get("primitives", ())]
    allowed = set(PRIMITIVE_ORDER)
    checks = {
        "strict_unit_array": valid_array,
        "exactly_nine_units": len(units) == 9,
        "required_unit_fields": all(isinstance(unit.get("id"), str) and isinstance(unit.get("name"), str) and isinstance(unit.get("definition"), str) and isinstance(unit.get("primitives"), (list, tuple)) for unit in units),
        "unique_unit_ids": len({unit.get("id") for unit in units}) == len(units) and all(unit.get("id") for unit in units),
        "primitive_coverage_exactly_once": set(memberships) == allowed and len(memberships) == len(PRIMITIVE_ORDER),
        "no_unknown_primitives": not (set(memberships) - allowed),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"unit mapping validation failed: {failed}")
    return candidate, checks


def compile_mapping(value: str | Mapping[str, Any]) -> dict[str, Any]:
    candidate, checks = validate_units(value)
    primitive_index = {name: index for index, name in enumerate(PRIMITIVE_ORDER)}
    compiled_units = []
    membership = []
    for unit in candidate["units"]:
        primitives = tuple(unit["primitives"])
        row = [1 if primitive in primitives else 0 for primitive in PRIMITIVE_ORDER]
        fields = {
            dataset: sorted({field for primitive in primitives for field in mapping[primitive]})
            for dataset, mapping in DATASET_PRIMITIVE_FIELDS.items()
        }
        compiled_units.append({**unit, "primitives": primitives, "dataset_raw_fields": fields})
        membership.append(row)
    if any(sum(row[index] for row in membership) != 1 for index in range(len(primitive_index))):
        raise ValueError("compiled membership differs from the primitive coverage contract")
    output_hash = canonical_hash(candidate)
    return {
        "schema_name": "fiesl_raw_fields_to_units",
        "schema_version": 1,
        "mapping_id": f"fiesl-nine-units-{output_hash[:12]}",
        "prompt_sha256": hashlib.sha256(build_prompt().encode("utf-8")).hexdigest(),
        "agent_output_sha256": output_hash,
        "primitive_order": PRIMITIVE_ORDER,
        "units": compiled_units,
        "training_membership": membership,
        "audit": {"status": "PASS", "checks": checks},
    }


def call_agent(endpoint: str, model: str, api_key: str, timeout_seconds: int) -> dict[str, Any]:
    if not endpoint or not model or not api_key:
        raise ValueError("endpoint, model, and API key are required")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt()}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("endpoint response contains no chat completion") from error
    return compile_mapping(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or replay the dataset raw-fields to nine-unit mapping.")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--response-file", type=Path)
    parser.add_argument("--call-agent", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.print_prompt:
        print(build_prompt())
        return
    if args.response_file is not None and args.call_agent:
        raise ValueError("choose either --response-file or --call-agent")
    if args.response_file is not None:
        result = compile_mapping(args.response_file.read_text(encoding="utf-8"))
    elif args.call_agent:
        result = call_agent(
            os.environ.get("FIESL_LLM_ENDPOINT", ""),
            os.environ.get("FIESL_LLM_MODEL", ""),
            os.environ.get("FIESL_LLM_API_KEY", ""),
            args.timeout_seconds,
        )
    else:
        result = compile_mapping({"units": FROZEN_UNITS})
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
