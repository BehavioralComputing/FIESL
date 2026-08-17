from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import torch


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

SOURCE_SLOT = (0, 0, 1, 1, 3, 4, 5, 6, 7, 8)

PRIMITIVES = (
    {"id": "identity_lexical", "raw_fields": ("name", "screen_name"), "source_slot": "Identity", "encoder": "shared_text_projection"},
    {"id": "identity_metadata", "raw_fields": ("protected", "verified", "has_extended_profile", "default_profile", "default_profile_image"), "source_slot": "Identity", "encoder": "shared_numeric_projection"},
    {"id": "profile_text", "raw_fields": ("description", "location", "url", "description_length", "location_length"), "source_slot": "Profile", "encoder": "shared_text_projection"},
    {"id": "profile_completeness", "raw_fields": ("description_present", "location_present", "url_present"), "source_slot": "Profile", "encoder": "shared_numeric_projection"},
    {"id": "popularity", "raw_fields": ("followers_count", "listed_count"), "source_slot": "Popularity", "encoder": "shared_numeric_projection"},
    {"id": "social_ratio", "raw_fields": ("followers_count", "friends_count"), "source_slot": "Social Ratio", "encoder": "shared_numeric_projection"},
    {"id": "activity_intensity", "raw_fields": ("statuses_count", "favourites_count"), "source_slot": "Activity Intensity", "encoder": "shared_numeric_projection"},
    {"id": "content_semantics", "raw_fields": ("own_tweets.text",), "source_slot": "Content Semantics", "encoder": "shared_text_projection"},
    {"id": "content_diversity", "raw_fields": ("own_tweets.text",), "source_slot": "Content Diversity", "encoder": "shared_numeric_projection"},
    {"id": "linguistic_style", "raw_fields": ("own_tweets.text",), "source_slot": "Linguistic Style", "encoder": "shared_numeric_projection"},
)

PROHIBITED_SOURCES = (
    "observed_account_topology",
    "external_neighbors",
    "account_id_value",
    "domain_annotation",
    "labels_as_features",
    "dev_metrics_as_features",
    "test_values",
    "relation_derived_features",
    "tweet_timestamps",
)

INACTIVE_SOURCES = ("Account Maturity", "Temporal Regularity")

DATASET_PROFILES = {
    "TwiBot-20": {
        "raw_files": ("train.json", "dev.json", "test.json"),
        "record_paths": {
            "identity_lexical": ("profile.name", "profile.screen_name"),
            "identity_metadata": ("profile.protected", "profile.verified", "profile.has_extended_profile", "profile.default_profile", "profile.default_profile_image"),
            "profile_text": ("profile.description", "profile.location", "profile.url"),
            "popularity": ("profile.followers_count", "profile.listed_count"),
            "social_ratio": ("profile.followers_count", "profile.friends_count"),
            "activity_intensity": ("profile.statuses_count", "profile.favourites_count"),
            "own_tweets": ("tweet[]",),
        },
        "unavailable_normalized_fields": (),
        "tweet_contract": "all own tweets in official record order",
    },
    "TwiBot-22": {
        "raw_files": ("user.json", "tweet_0.json", "tweet_1.json", "tweet_2.json", "tweet_3.json", "tweet_4.json", "tweet_5.json", "tweet_6.json", "tweet_7.json", "tweet_8.json"),
        "record_paths": {
            "identity_lexical": ("user.name", "user.username"),
            "identity_metadata": ("user.protected", "user.verified"),
            "profile_text": ("user.description", "user.location", "user.url"),
            "popularity": ("user.public_metrics.followers_count", "user.public_metrics.listed_count"),
            "social_ratio": ("user.public_metrics.followers_count", "user.public_metrics.following_count"),
            "activity_intensity": ("user.public_metrics.tweet_count",),
            "own_tweets": ("tweet_*.text joined by documented author_id",),
        },
        "unavailable_normalized_fields": ("has_extended_profile", "default_profile", "default_profile_image", "favourites_count"),
        "tweet_contract": "first 20 own tweets in official dataset order",
    },
}


def dataset_profiles_payload(datasets: tuple[str, ...] = ("TwiBot-20", "TwiBot-22")) -> dict[str, Any]:
    unknown = sorted(set(datasets) - set(DATASET_PROFILES))
    if unknown:
        raise ValueError(f"unknown dataset profiles: {unknown}")
    return {name: DATASET_PROFILES[name] for name in datasets}


def build_agent_prompt(datasets: tuple[str, ...] = ("TwiBot-20", "TwiBot-22")) -> str:
    profiles = json.dumps(dataset_profiles_payload(datasets), ensure_ascii=False, indent=2, sort_keys=True)
    return f"""You are a dataset-level semantic evidence schema compiler. Return exactly one JSON schema containing 6-12 units. You may only group the ten supplied primitives; you may not add, transform, inspect, or infer account-level values.

Allowed primitives:
identity_lexical, identity_metadata, profile_text, profile_completeness, popularity, social_ratio, activity_intensity, content_semantics, content_diversity, linguistic_style.

Constraints:
- Cover every primitive exactly once.
- Use no label, Dev/Test metric, historical model outcome, external account, observed topology, neighbor, account identifier value, domain annotation, timestamp, or relation-derived feature.
- Preserve own-Tweet semantics, diversity, and style as declared views of one source; do not duplicate any raw feature.
- Prefer cross-dataset semantic clarity and avoid tiny units unless separating them is semantically necessary.
- Output unit id, name, definition, and primitive membership only.
- Do not propose edges, encoders, transforms, code, or per-account decisions.

Generation policy: one output only. The exact output is frozen by SHA-256 and must never be edited after seeing training or Dev results.

The following is field-name metadata only. It contains no account records, labels, graph files, or metrics. The deterministic preprocessing adapter normalizes these fields before feature construction:
{profiles}
"""


LLM_SCHEMA_PROMPT = build_agent_prompt()

FROZEN_LLM_OUTPUT = {
    "candidate_id": "S-Agent",
    "origin": "single_frozen_dataset_level_llm_output",
    "compiler_provenance": {
        "role": "dataset-level LLM schema compiler",
        "provider": "withheld_for_anonymous_review",
        "model_family": "general_purpose_llm",
        "generation_count": 1,
        "account_level_calls": 0,
        "regeneration_after_metrics": False,
    },
    "units": (
        {"id": "identity_lexical", "name": "Identity Lexical", "definition": "Lexical and character-form identity presentation.", "primitives": ("identity_lexical",)},
        {"id": "profile_text", "name": "Profile Text", "definition": "Free-text profile self-description and its length form.", "primitives": ("profile_text",)},
        {"id": "presentation_metadata", "name": "Presentation Metadata", "definition": "Low-dimensional identity flags and profile completeness indicators.", "primitives": ("identity_metadata", "profile_completeness")},
        {"id": "popularity", "name": "Popularity", "definition": "Follower and list-based social standing.", "primitives": ("popularity",)},
        {"id": "social_ratio", "name": "Social Ratio", "definition": "Follower/friend balance distinct from absolute standing.", "primitives": ("social_ratio",)},
        {"id": "activity", "name": "Activity", "definition": "Account-local activity intensity.", "primitives": ("activity_intensity",)},
        {"id": "content_semantics", "name": "Content Semantics", "definition": "Semantic summary of own Tweets.", "primitives": ("content_semantics",)},
        {"id": "content_diversity", "name": "Content Diversity", "definition": "Dispersion and repetition of own Tweets.", "primitives": ("content_diversity",)},
        {"id": "linguistic_style", "name": "Linguistic Style", "definition": "Label-free linguistic form of own Tweets.", "primitives": ("linguistic_style",)},
    ),
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_agent_output(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        text = value.strip()
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("agent response contains no JSON object")
            text = text[start : end + 1]
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("agent response must be a JSON object")
    return parsed


def audit_agent_output(value: str | Mapping[str, Any]) -> dict[str, Any]:
    candidate = parse_agent_output(value)
    units = candidate.get("units")
    valid_units = isinstance(units, (list, tuple)) and all(isinstance(unit, Mapping) for unit in units)
    units = list(units) if valid_units else []
    memberships = [primitive for unit in units for primitive in unit.get("primitives", ())]
    primitive_set = set(PRIMITIVE_ORDER)
    checks = {
        "strict_unit_array": valid_units,
        "unit_count_6_to_12": 6 <= len(units) <= 12,
        "unit_fields": all(isinstance(unit.get("id"), str) and isinstance(unit.get("name"), str) and isinstance(unit.get("definition"), str) and isinstance(unit.get("primitives"), (list, tuple)) for unit in units),
        "unique_unit_ids": len({unit.get("id") for unit in units}) == len(units) and all(unit.get("id") for unit in units),
        "primitive_coverage_exactly_once": set(memberships) == primitive_set and len(memberships) == len(PRIMITIVE_ORDER),
        "no_unknown_primitives": not (set(memberships) - primitive_set),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate": candidate,
    }


def candidate_manifest(value: str | Mapping[str, Any], datasets: tuple[str, ...] = ("TwiBot-20", "TwiBot-22")) -> dict[str, Any]:
    audit = audit_agent_output(value)
    if audit["status"] != "PASS":
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise ValueError(f"agent schema audit failed: {failed}")
    candidate = audit["candidate"]
    return {
        "schema_name": "fiesl_agent_candidate",
        "schema_version": 1,
        "datasets": datasets,
        "dataset_profiles_sha256": canonical_hash(dataset_profiles_payload(datasets)),
        "agent_prompt_sha256": hashlib.sha256(build_agent_prompt(datasets).encode("utf-8")).hexdigest(),
        "agent_output_sha256": canonical_hash(candidate),
        "units": candidate["units"],
        "audit": {"status": audit["status"], "checks": audit["checks"]},
        "candidate_only": True,
    }


def call_compatible_endpoint(prompt: str, endpoint: str, model: str, api_key: str, timeout_seconds: int = 90) -> dict[str, Any]:
    if not endpoint or not model or not api_key:
        raise ValueError("endpoint, model, and API key are required for an agent call")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
        raise ValueError("endpoint response does not contain a chat completion") from error
    if not isinstance(content, str):
        raise ValueError("endpoint completion content must be a JSON string")
    return parse_agent_output(content)


def run_agent_from_environment(datasets: tuple[str, ...] = ("TwiBot-20", "TwiBot-22"), timeout_seconds: int = 90) -> dict[str, Any]:
    endpoint = os.environ.get("FIESL_LLM_ENDPOINT", "")
    model = os.environ.get("FIESL_LLM_MODEL", "")
    api_key = os.environ.get("FIESL_LLM_API_KEY", "")
    output = call_compatible_endpoint(build_agent_prompt(datasets), endpoint, model, api_key, timeout_seconds)
    return candidate_manifest(output, datasets)


def compile_ontology() -> dict[str, Any]:
    primitive_ids = tuple(item["id"] for item in PRIMITIVES)
    units = FROZEN_LLM_OUTPUT["units"]
    memberships = tuple(primitive for unit in units for primitive in unit["primitives"])
    agent_audit = audit_agent_output(FROZEN_LLM_OUTPUT)
    checks = {
        "primitive_order_matches_contract": primitive_ids == PRIMITIVE_ORDER,
        "ten_primitives": len(PRIMITIVE_ORDER) == 10,
        "nine_units": len(units) == 9,
        "unique_unit_ids": len({unit["id"] for unit in units}) == len(units),
        "primitive_coverage_exactly_once": set(memberships) == set(PRIMITIVE_ORDER) and len(memberships) == len(PRIMITIVE_ORDER),
        "account_level_calls": FROZEN_LLM_OUTPUT["compiler_provenance"]["account_level_calls"] == 0,
        "single_generation": FROZEN_LLM_OUTPUT["compiler_provenance"]["generation_count"] == 1,
        "no_metric_informed_regeneration": FROZEN_LLM_OUTPUT["compiler_provenance"]["regeneration_after_metrics"] is False,
        "prohibited_sources_absent": not set().union(*(set(item["raw_fields"]) for item in PRIMITIVES)).intersection(PROHIBITED_SOURCES),
        "cross_dataset_raw_schema_present": set(dataset_profiles_payload()) == {"TwiBot-20", "TwiBot-22"},
        "frozen_agent_output_valid": agent_audit["status"] == "PASS",
    }
    if not all(checks.values()):
        raise ValueError(f"frozen ontology audit failed: {[name for name, value in checks.items() if not value]}")
    primitive_contract = {
        "primitive_order": PRIMITIVE_ORDER,
        "primitives": PRIMITIVES,
        "prohibited_sources": PROHIBITED_SOURCES,
        "inactive_sources": INACTIVE_SOURCES,
        "feature_mapping_policy": {
            "text_projection": "shared LayerNorm-MLP",
            "numeric_projection": "shared LayerNorm-MLP",
            "quality_projection": "shared LayerNorm-linear map",
            "intra_unit_pooling": "masked normalized mean",
        },
    }
    ontology_hash = canonical_hash({"primitive_contract": primitive_contract, "frozen_output": FROZEN_LLM_OUTPUT})
    return {
        "schema_name": "fiesl_frozen_llm_ontology",
        "schema_version": 1,
        "evidence_ontology_id": f"fiesl-s-agent-v1-{ontology_hash[:12]}",
        "ontology_sha256": ontology_hash,
        "primitive_contract_sha256": canonical_hash(primitive_contract),
        "agent_prompt_sha256": hashlib.sha256(LLM_SCHEMA_PROMPT.encode("utf-8")).hexdigest(),
        "agent_output_sha256": canonical_hash(FROZEN_LLM_OUTPUT),
        "primitive_order": PRIMITIVE_ORDER,
        "units": units,
        "unit_count": len(units),
        "dataset_profiles": dataset_profiles_payload(),
        "dataset_profiles_sha256": canonical_hash(dataset_profiles_payload()),
        "primitive_contract": primitive_contract,
        "compiler_provenance": FROZEN_LLM_OUTPUT["compiler_provenance"],
        "audit": {"status": "PASS", "checks": checks},
    }


def unit_membership(manifest: dict[str, Any] | None = None) -> torch.Tensor:
    resolved = compile_ontology() if manifest is None else manifest
    index = {name: position for position, name in enumerate(resolved["primitive_order"])}
    membership = torch.zeros((len(resolved["units"]), len(index)), dtype=torch.float32)
    for unit_index, unit in enumerate(resolved["units"]):
        for primitive in unit["primitives"]:
            membership[unit_index, index[primitive]] = 1.0
    return membership


def write_ontology_manifest(output_path: Path) -> dict[str, Any]:
    manifest = compile_ontology()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, replay, or audit the frozen FIESL evidence ontology.")
    parser.add_argument("--dataset", action="append", choices=tuple(DATASET_PROFILES))
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--response-file", type=Path)
    parser.add_argument("--call-agent", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets = tuple(args.dataset) if args.dataset else ("TwiBot-20", "TwiBot-22")
    if args.print_prompt:
        print(build_agent_prompt(datasets))
        return
    if args.response_file is not None and args.call_agent:
        raise ValueError("choose either --response-file or --call-agent")
    if args.response_file is not None:
        manifest = candidate_manifest(args.response_file.read_text(encoding="utf-8"), datasets)
    elif args.call_agent:
        manifest = run_agent_from_environment(datasets, args.timeout_seconds)
    else:
        manifest = compile_ontology()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
