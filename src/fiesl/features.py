from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable

import torch
import torch.nn.functional as functional

from fiesl.encoding import TextEncoder
from fiesl.raw import Account


EVIDENCE_ORDER = (
    "Identity",
    "Profile",
    "Account Maturity",
    "Popularity",
    "Social Ratio",
    "Activity Intensity",
    "Content Semantics",
    "Content Diversity",
    "Linguistic Style",
)
INPUT_DIMS = (783, 773, 1, 4, 4, 4, 768, 9, 15)
QUALITY_ORDER = (
    "observed_fraction",
    "log1p_sample_count",
    "log1p_total_character_count",
    "log1p_average_character_count",
    "exact_duplicate_ratio",
    "missing_fraction",
    "is_text_evidence",
    "is_numeric_evidence",
)
NUMERIC_ORDER = (
    "followers_count_log1p",
    "listed_count_log1p",
    "followers_friends_log_ratio",
    "followers_share",
    "statuses_count_log1p",
    "favourites_count_log1p",
)
STYLE_ORDER = (
    "log1p_tweet_count",
    "log1p_total_character_count",
    "log1p_average_character_count",
    "log1p_average_token_count",
    "url_token_ratio",
    "hashtag_token_ratio",
    "mention_token_ratio",
    "uppercase_letter_ratio",
    "digit_character_ratio",
    "punctuation_character_ratio",
    "non_ascii_character_ratio",
    "lexical_diversity",
    "exact_duplicate_ratio",
    "whitespace_character_ratio",
    "line_break_character_ratio",
)
TOKEN_PATTERN = re.compile(r"\S+", flags=re.UNICODE)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


class RobustScaler:
    def __init__(self, names: Iterable[str], epsilon: float = 1e-6) -> None:
        self.names = tuple(names)
        self.epsilon = float(epsilon)
        self.statistics: dict[str, dict[str, float | int | bool]] = {}

    def fit(self, rows: Iterable[dict[str, float | None]]) -> "RobustScaler":
        observed = {name: [] for name in self.names}
        count = 0
        for row in rows:
            count += 1
            for name in self.names:
                value = row.get(name)
                if value is not None:
                    observed[name].append(float(value))
        if count == 0:
            raise ValueError("Cannot fit preprocessing without Train records")
        for name in self.names:
            values = observed[name]
            median = quantile(values, 0.5)
            first = quantile(values, 0.25)
            third = quantile(values, 0.75)
            self.statistics[name] = {
                "count": len(values),
                "missing_count": count - len(values),
                "median": median,
                "q1": first,
                "q3": third,
                "iqr": third - first,
                "scale_denominator": max(third - first, self.epsilon),
                "all_train_values_missing": not values,
            }
        return self

    def transform(self, row: dict[str, float | None]) -> tuple[list[float], list[float]]:
        values = []
        missing = []
        for name in self.names:
            raw = row.get(name)
            state = self.statistics[name]
            is_missing = raw is None
            imputed = float(state["median"]) if is_missing else float(raw)
            values.append((imputed - float(state["median"])) / float(state["scale_denominator"]))
            missing.append(float(is_missing))
        return values, missing

    def state_dict(self) -> dict[str, Any]:
        return {
            "fit_split": "train",
            "epsilon": self.epsilon,
            "feature_order": list(self.names),
            "statistics": self.statistics,
        }


def numeric_features(values: dict[str, Any]) -> dict[str, float | None]:
    followers = values.get("followers_count")
    friends = values.get("friends_count")
    listed = values.get("listed_count")
    statuses = values.get("statuses_count")
    favourites = values.get("favourites_count")
    ratio = math.log((followers + 1) / (friends + 1)) if followers is not None and friends is not None else None
    share = (followers + 1) / (followers + friends + 2) if followers is not None and friends is not None else None
    return {
        "followers_count_log1p": math.log1p(followers) if followers is not None else None,
        "listed_count_log1p": math.log1p(listed) if listed is not None else None,
        "followers_friends_log_ratio": ratio,
        "followers_share": share,
        "statuses_count_log1p": math.log1p(statuses) if statuses is not None else None,
        "favourites_count_log1p": math.log1p(favourites) if favourites is not None else None,
    }


def style_features(texts: list[str]) -> dict[str, float] | None:
    if not texts:
        return None
    combined = "\n".join(texts)
    characters = list(combined)
    tokens = [token for text in texts for token in TOKEN_PATTERN.findall(text)]
    token_count = len(tokens)
    character_count = len(characters)
    alpha_count = sum(character.isalpha() for character in characters)

    def token_ratio(prefixes: tuple[str, ...]) -> float:
        return 0.0 if token_count == 0 else sum(token.casefold().startswith(prefixes) for token in tokens) / token_count

    def character_ratio(predicate: Any) -> float:
        return 0.0 if character_count == 0 else sum(predicate(character) for character in characters) / character_count

    return {
        "log1p_tweet_count": math.log1p(len(texts)),
        "log1p_total_character_count": math.log1p(sum(map(len, texts))),
        "log1p_average_character_count": math.log1p(sum(map(len, texts)) / len(texts)),
        "log1p_average_token_count": math.log1p(token_count / len(texts)),
        "url_token_ratio": token_ratio(("http://", "https://", "www.")),
        "hashtag_token_ratio": token_ratio(("#",)),
        "mention_token_ratio": token_ratio(("@",)),
        "uppercase_letter_ratio": 0.0 if alpha_count == 0 else sum(character.isupper() for character in characters) / alpha_count,
        "digit_character_ratio": character_ratio(str.isdigit),
        "punctuation_character_ratio": character_ratio(lambda character: unicodedata.category(character).startswith("P")),
        "non_ascii_character_ratio": character_ratio(lambda character: ord(character) > 127),
        "lexical_diversity": 0.0 if token_count == 0 else len({token.casefold() for token in tokens}) / token_count,
        "exact_duplicate_ratio": 1 - len(set(texts)) / len(texts),
        "whitespace_character_ratio": character_ratio(str.isspace),
        "line_break_character_ratio": character_ratio(lambda character: character in {"\n", "\r"}),
    }


def identity_inputs(values: dict[str, Any]) -> tuple[str, list[float]]:
    display = values.get("name") or ""
    screen = values.get("screen_name") or ""

    def ratio(text: str, predicate: Any) -> float:
        return 0.0 if not text else sum(predicate(character) for character in text) / len(text)

    flags = [values.get(name) for name in ("protected", "verified", "has_extended_profile", "default_profile", "default_profile_image")]
    features = [
        math.log1p(len(display)),
        math.log1p(len(screen)),
        ratio(screen, str.isalpha),
        ratio(screen, str.isdigit),
        ratio(screen, lambda character: character == "_"),
        ratio(screen, lambda character: not (character.isalnum() or character == "_")),
        ratio(display, str.isalpha),
        ratio(display, str.isdigit),
        ratio(display, str.isspace),
        ratio(display, lambda character: ord(character) > 127),
        *[0.0 if value is None else float(bool(value)) for value in flags],
    ]
    return display, features


def profile_inputs(values: dict[str, Any]) -> tuple[str, list[float]]:
    description = values.get("description") or ""
    location = values.get("location") or ""
    url = values.get("url") or ""
    parts = [text for text in (f"Description: {description}" if description else "", f"Location: {location}" if location else "") if text]
    return "\n".join(parts), [float(bool(description)), float(bool(location)), float(bool(url)), math.log1p(len(description)), math.log1p(len(location))]


def diversity_features(embeddings: torch.Tensor, texts: list[str]) -> torch.Tensor:
    count = len(texts)
    if count == 0 or embeddings.shape[0] != count:
        raise ValueError("Content diversity requires aligned nonempty tweets")
    variance = embeddings.var(dim=0, unbiased=False)
    normalized = functional.normalize(embeddings, dim=1, eps=1e-12)
    centroid = functional.normalize(embeddings.mean(dim=0, keepdim=True), dim=1, eps=1e-12)
    centroid_distance = 1 - (normalized * centroid).sum(dim=1)
    pairs = []
    for left in range(count):
        for right in range(left + 1, count):
            pairs.append(1 - (normalized[left] * normalized[right]).sum())
            if len(pairs) == 1024:
                break
        if len(pairs) == 1024:
            break
    distances = torch.stack(pairs) if pairs else torch.zeros(1)
    return torch.tensor(
        [
            float(variance.mean()),
            float(variance.std(unbiased=False)),
            float(variance.max()),
            float(centroid_distance.mean()),
            float(distances.mean()),
            float(distances.std(unbiased=False)),
            float(distances.max()),
            1 - len(set(texts)) / count,
            math.log1p(count),
        ],
        dtype=torch.float32,
    )


def quality_vector(name: str, observed: int, total: int, texts: list[str] | None = None) -> torch.Tensor:
    if texts is not None:
        count = len(texts)
        characters = sum(map(len, texts))
        average = characters / count if count else 0.0
        duplicate = 1 - len(set(texts)) / count if count else 0.0
        fraction = float(count > 0)
        text_flag = 1.0
        numeric_flag = 0.0
    else:
        count = observed
        characters = 0
        average = 0.0
        duplicate = 0.0
        fraction = observed / total if total else 0.0
        text_flag = float(name in {"Identity", "Profile"})
        numeric_flag = float(name in {"Popularity", "Social Ratio", "Activity Intensity"})
    return torch.tensor([fraction, math.log1p(count), math.log1p(characters), math.log1p(average), duplicate, 1 - fraction, text_flag, numeric_flag], dtype=torch.float32)


def fit_preprocessors(accounts: Iterable[Account]) -> tuple[RobustScaler, RobustScaler]:
    numeric_rows = []
    style_rows = []
    for account in accounts:
        if account.split != "train":
            raise ValueError("Preprocessing may only fit on Train")
        numeric_rows.append(numeric_features(account.values))
        style = style_features(account.tweets)
        if style is not None:
            style_rows.append(style)
    numeric = RobustScaler(NUMERIC_ORDER).fit(numeric_rows)
    style = RobustScaler(STYLE_ORDER).fit(style_rows)
    return numeric, style


def build_payload(accounts: list[Account], encoder: TextEncoder, numeric: RobustScaler, style: RobustScaler, batch_size: int) -> dict[str, Any]:
    if encoder.dimension != 768:
        raise ValueError("The FIESL evidence contract requires 768-dimensional text vectors")
    count = len(accounts)
    typed = torch.zeros((count, 9, 783), dtype=torch.float32)
    availability = torch.zeros((count, 9), dtype=torch.bool)
    quality = torch.zeros((count, 9, 8), dtype=torch.float32)
    identity = [identity_inputs(account.values) for account in accounts]
    profile = [profile_inputs(account.values) for account in accounts]
    identity_vectors = encoder.encode([item[0] for item in identity], batch_size)
    profile_indices = [index for index, account in enumerate(accounts) if any(account.values.get(name) for name in ("description", "location", "url"))]
    profile_vectors = torch.zeros((count, 768), dtype=torch.float32)
    if profile_indices:
        encoded_profiles = encoder.encode([profile[index][0] for index in profile_indices], batch_size)
        profile_vectors[torch.tensor(profile_indices)] = encoded_profiles
    flat_tweets = [text for account in accounts for text in account.tweets]
    tweet_vectors = encoder.encode(flat_tweets, batch_size)
    offset = 0
    for index, account in enumerate(accounts):
        values = account.values
        observed_identity = sum(values.get(name) is not None for name in ("name", "screen_name", "protected", "verified", "has_extended_profile", "default_profile", "default_profile_image"))
        observed_profile = sum(values.get(name) is not None for name in ("description", "location", "url"))
        availability[index, 0] = observed_identity > 0
        availability[index, 1] = observed_profile > 0
        typed[index, 0, :783] = torch.cat((identity_vectors[index], torch.tensor(identity[index][1])))
        typed[index, 1, :773] = torch.cat((profile_vectors[index], torch.tensor(profile[index][1])))
        quality[index, 0] = quality_vector("Identity", observed_identity, 7)
        quality[index, 1] = quality_vector("Profile", observed_profile, 3)
        raw_numeric = numeric_features(values)
        scaled, missing = numeric.transform(raw_numeric)
        for slot, indices, name in ((3, (0, 1), "Popularity"), (4, (2, 3), "Social Ratio"), (5, (4, 5), "Activity Intensity")):
            vector = [scaled[item] for item in indices] + [missing[item] for item in indices]
            typed[index, slot, :4] = torch.tensor(vector)
            observed = sum(not bool(missing[item]) for item in indices)
            availability[index, slot] = observed > 0
            quality[index, slot] = quality_vector(name, observed, 2)
        texts = account.tweets
        if texts:
            embeddings = tweet_vectors[offset : offset + len(texts)]
            typed[index, 6, :768] = embeddings.mean(dim=0)
            typed[index, 7, :9] = diversity_features(embeddings, texts)
            raw_style = style_features(texts)
            style_values, _ = style.transform(raw_style or {})
            typed[index, 8, :15] = torch.tensor(style_values)
            availability[index, 6:9] = True
            for slot, name in ((6, "Content Semantics"), (7, "Content Diversity"), (8, "Linguistic Style")):
                quality[index, slot] = quality_vector(name, len(texts), len(texts), texts)
        offset += len(texts)
    availability[:, 2] = False
    typed *= availability.unsqueeze(-1)
    quality *= availability.unsqueeze(-1)
    return {
        "account_ids": [account.account_id for account in accounts],
        "labels": torch.tensor([account.label for account in accounts], dtype=torch.int64),
        "typed_inputs": typed,
        "input_dims": torch.tensor(INPUT_DIMS, dtype=torch.int64),
        "availability_mask": availability,
        "quality_features": quality,
    }
