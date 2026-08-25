from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
FROZEN_COMPILATION_PATH = ROOT / "configs" / "frozen_schema_compilation.json"

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

DATASET_RAW_SCHEMAS = {
    "TwiBot-20": (
        {"field": "profile.name", "type": "string", "description": "Focal-account display name."},
        {"field": "profile.screen_name", "type": "string", "description": "Focal-account screen name."},
        {"field": "profile.protected", "type": "boolean", "description": "Protected-account flag."},
        {"field": "profile.verified", "type": "boolean", "description": "Verified-account flag."},
        {"field": "profile.has_extended_profile", "type": "boolean", "description": "Extended-profile flag."},
        {"field": "profile.default_profile", "type": "boolean", "description": "Default-profile flag."},
        {"field": "profile.default_profile_image", "type": "boolean", "description": "Default-profile-image flag."},
        {"field": "profile.description", "type": "string", "description": "Focal-account self-description."},
        {"field": "profile.location", "type": "string", "description": "Self-declared profile location."},
        {"field": "profile.url", "type": "string", "description": "Profile URL text."},
        {"field": "profile.followers_count", "type": "count", "description": "Follower count."},
        {"field": "profile.friends_count", "type": "count", "description": "Following or friend count."},
        {"field": "profile.listed_count", "type": "count", "description": "Public-list membership count."},
        {"field": "profile.statuses_count", "type": "count", "description": "Published-status count."},
        {"field": "profile.favourites_count", "type": "count", "description": "Favorite-action count."},
        {"field": "tweet[]", "type": "text collection", "description": "Text of the focal account's own tweets."},
    ),
    "TwiBot-22": (
        {"field": "user.name", "type": "string", "description": "Focal-account display name."},
        {"field": "user.username", "type": "string", "description": "Focal-account username."},
        {"field": "user.protected", "type": "boolean", "description": "Protected-account flag."},
        {"field": "user.verified", "type": "boolean", "description": "Verified-account flag."},
        {"field": "user.description", "type": "string", "description": "Focal-account self-description."},
        {"field": "user.location", "type": "string", "description": "Self-declared profile location."},
        {"field": "user.url", "type": "string", "description": "Profile URL text."},
        {"field": "user.public_metrics.followers_count", "type": "count", "description": "Follower count."},
        {"field": "user.public_metrics.following_count", "type": "count", "description": "Following count."},
        {"field": "user.public_metrics.listed_count", "type": "count", "description": "Public-list membership count."},
        {"field": "user.public_metrics.tweet_count", "type": "count", "description": "Published-tweet count."},
        {"field": "tweet_*.json.text", "type": "text collection", "description": "Text of the focal account's own tweets."},
    ),
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt() -> str:
    schemas = json.dumps(DATASET_RAW_SCHEMAS, ensure_ascii=False, indent=2, sort_keys=True)
    primitives = ", ".join(PRIMITIVE_ORDER)
    return f"""You are a constrained dataset-level semantic evidence schema compiler. Jointly align the raw-field schemas of TwiBot-20 and TwiBot-22 to a shared primitive vocabulary and organize those primitives into exactly nine semantic evidence units.

Allowed shared primitives:
{primitives}

Return exactly one JSON object with:
1. field_alignments: one array for each dataset. Every array item must contain a field and a non-empty primitives array.
2. units: exactly nine objects, each containing id, name, definition, and primitives.

Requirements:
- Include every listed raw field exactly once in its dataset's field_alignments array.
- Assign only allowed primitives. A raw field may support multiple primitives when it provides distinct semantic views, such as profile text and completeness, popularity and social ratio, or content semantics, diversity, and style.
- Use every allowed primitive for both datasets.
- Cover every allowed primitive exactly once across the nine units.
- Use schema metadata only. Never inspect account values or identifiers.
- Never use labels, Train/Dev/Test metrics, model results, graph edges, neighbors, timestamps, domain annotations, or relation-derived features.
- Do not propose encoders, transforms, edges, hyperparameters, or per-account decisions.

Dataset raw-field metadata:
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


def validate_compilation(value: str | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate = parse_response(value)
    field_alignments = candidate.get("field_alignments")
    strict_alignment_object = isinstance(field_alignments, Mapping) and set(field_alignments) == set(DATASET_RAW_SCHEMAS)
    alignments = dict(field_alignments) if strict_alignment_object else {}
    allowed = set(PRIMITIVE_ORDER)
    expected_fields = {
        dataset: {item["field"] for item in schema}
        for dataset, schema in DATASET_RAW_SCHEMAS.items()
    }
    alignment_arrays = all(
        isinstance(alignments.get(dataset), (list, tuple))
        and all(isinstance(item, Mapping) for item in alignments[dataset])
        for dataset in DATASET_RAW_SCHEMAS
    )
    required_alignment_fields = alignment_arrays and all(
        isinstance(item.get("field"), str)
        and isinstance(item.get("primitives"), (list, tuple))
        and bool(item.get("primitives"))
        and all(isinstance(primitive, str) for primitive in item["primitives"])
        for dataset in DATASET_RAW_SCHEMAS
        for item in alignments[dataset]
    )
    exact_field_coverage = required_alignment_fields and all(
        len([item["field"] for item in alignments[dataset]]) == len(expected_fields[dataset])
        and {item["field"] for item in alignments[dataset]} == expected_fields[dataset]
        for dataset in DATASET_RAW_SCHEMAS
    )
    alignment_primitives_allowed = required_alignment_fields and all(
        not (set(item["primitives"]) - allowed)
        and len(item["primitives"]) == len(set(item["primitives"]))
        for dataset in DATASET_RAW_SCHEMAS
        for item in alignments[dataset]
    )
    all_primitives_supported_per_dataset = alignment_primitives_allowed and all(
        {
            primitive
            for item in alignments[dataset]
            for primitive in item["primitives"]
        }
        == allowed
        for dataset in DATASET_RAW_SCHEMAS
    )

    units_value = candidate.get("units")
    strict_unit_array = isinstance(units_value, (list, tuple)) and all(
        isinstance(unit, Mapping) for unit in units_value
    )
    units = list(units_value) if strict_unit_array else []
    required_unit_fields = all(
        isinstance(unit.get("id"), str)
        and isinstance(unit.get("name"), str)
        and isinstance(unit.get("definition"), str)
        and isinstance(unit.get("primitives"), (list, tuple))
        for unit in units
    )
    memberships = [
        primitive
        for unit in units
        if isinstance(unit.get("primitives"), (list, tuple))
        for primitive in unit["primitives"]
    ]
    checks = {
        "strict_field_alignment_object": strict_alignment_object,
        "strict_field_alignment_arrays": alignment_arrays,
        "required_field_alignment_fields": required_alignment_fields,
        "raw_field_coverage_exactly_once": exact_field_coverage,
        "alignment_primitives_allowed": alignment_primitives_allowed,
        "all_primitives_supported_per_dataset": all_primitives_supported_per_dataset,
        "strict_unit_array": strict_unit_array,
        "exactly_nine_units": len(units) == 9,
        "required_unit_fields": required_unit_fields,
        "unique_unit_ids": len({unit.get("id") for unit in units}) == len(units)
        and all(unit.get("id") for unit in units),
        "primitive_coverage_exactly_once": set(memberships) == allowed
        and len(memberships) == len(PRIMITIVE_ORDER),
        "no_unknown_unit_primitives": not (set(memberships) - allowed),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"schema compilation validation failed: {failed}")
    return candidate, checks


def compile_mapping(
    value: str | Mapping[str, Any],
    compiler_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate, checks = validate_compilation(value)
    primitive_fields: dict[str, dict[str, list[str]]] = {
        dataset: {primitive: [] for primitive in PRIMITIVE_ORDER}
        for dataset in DATASET_RAW_SCHEMAS
    }
    for dataset, alignments in candidate["field_alignments"].items():
        for alignment in alignments:
            for primitive in alignment["primitives"]:
                primitive_fields[dataset][primitive].append(alignment["field"])

    compiled_units = []
    membership = []
    primitive_index = {name: index for index, name in enumerate(PRIMITIVE_ORDER)}
    for unit in candidate["units"]:
        primitives = tuple(unit["primitives"])
        row = [1 if primitive in primitives else 0 for primitive in PRIMITIVE_ORDER]
        fields = {
            dataset: sorted(
                {
                    field
                    for primitive in primitives
                    for field in primitive_fields[dataset][primitive]
                }
            )
            for dataset in DATASET_RAW_SCHEMAS
        }
        compiled_units.append({**unit, "primitives": primitives, "dataset_raw_fields": fields})
        membership.append(row)
    if any(sum(row[index] for row in membership) != 1 for index in range(len(primitive_index))):
        raise ValueError("compiled membership differs from the primitive coverage contract")

    output_hash = canonical_hash(candidate)
    return {
        "schema_name": "fiesl_raw_fields_to_units",
        "schema_version": 2,
        "mapping_id": f"fiesl-nine-units-{output_hash[:12]}",
        "prompt_sha256": hashlib.sha256(build_prompt().encode("utf-8")).hexdigest(),
        "agent_output_sha256": output_hash,
        "compiler_provenance": dict(compiler_provenance or {}),
        "primitive_order": PRIMITIVE_ORDER,
        "field_alignments": candidate["field_alignments"],
        "units": compiled_units,
        "training_membership": membership,
        "audit": {"status": "PASS", "checks": checks},
    }


def load_frozen_compilation(path: Path = FROZEN_COMPILATION_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ValueError("frozen compilation must contain one response object")
    provenance = payload.get("compiler_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("compiler provenance must be an object")
    return payload["response"], provenance


def load_response_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
        provenance = payload.get("compiler_provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("compiler provenance must be an object")
        return payload["response"], provenance
    if not isinstance(payload, dict):
        raise ValueError("response file must contain a JSON object")
    return payload, {"mode": "external_response_replay"}


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
    return compile_mapping(
        content,
        {
            "mode": "live_schema_compilation",
            "model": model,
            "account_level_calls": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or replay the dataset raw-field to primitive to unit mapping."
    )
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
        response, provenance = load_response_file(args.response_file)
        result = compile_mapping(response, provenance)
    elif args.call_agent:
        result = call_agent(
            os.environ.get("FIESL_LLM_ENDPOINT", ""),
            os.environ.get("FIESL_LLM_MODEL", ""),
            os.environ.get("FIESL_LLM_API_KEY", ""),
            args.timeout_seconds,
        )
    else:
        response, provenance = load_frozen_compilation()
        result = compile_mapping(response, provenance)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
