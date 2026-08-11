from __future__ import annotations

import csv
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Account:
    account_id: str
    split: str
    label: int
    values: dict[str, Any]
    tweets: list[str]


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value.strip())
    if text.casefold() in {"", "none", "null", "n/a", "nan"}:
        return None
    text = "".join(" " if unicodedata.category(character) == "Cc" and character not in {"\n", "\r", "\t"} else character for character in text)
    return text if text.strip() else None


def nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 and number < float("inf") else None


def optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false", "0", "1"}:
        return value.strip().casefold() in {"true", "1"}
    return None


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            while position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = buffer == ""
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    raise ValueError(f"Empty JSON array: {path}")
                if buffer[position] != "[":
                    raise ValueError(f"Expected a top-level JSON array: {path}")
                position += 1
                started = True
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"Unterminated JSON array: {path}")
                buffer = handle.read(chunk_size)
                position = 0
                eof = buffer == ""
                continue
            if buffer[position] == "]":
                return
            start = position
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    yield value
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"Invalid JSON in {path}: {error}") from error
                    remainder = buffer[start:]
                    if len(remainder) > 64 << 20:
                        raise ValueError(f"Oversized JSON record in {path}")
                    chunk = handle.read(chunk_size)
                    eof = chunk == ""
                    buffer = remainder + chunk
                    position = 0
                    start = 0
            if position > chunk_size * 4:
                buffer = buffer[position:]
                position = 0


def normalize_account_id(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("Missing account ID")
    return text if text.startswith("u") else f"u{text}"


def canonical_split(value: Any) -> str:
    text = str(value).strip().casefold()
    text = "dev" if text in {"val", "valid", "validation"} else text
    if text not in {"train", "dev", "test"}:
        raise ValueError(f"Invalid split: {value!r}")
    return text


def canonical_label(value: Any) -> int:
    text = str(value).strip().casefold()
    if text in {"bot", "1"}:
        return 1
    if text in {"human", "0"}:
        return 0
    raise ValueError(f"Invalid label: {value!r}")


def twibot20_values(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
    return {
        "name": clean_text(profile.get("name")),
        "screen_name": clean_text(profile.get("screen_name")),
        "protected": optional_bool(profile.get("protected")),
        "verified": optional_bool(profile.get("verified")),
        "has_extended_profile": optional_bool(profile.get("has_extended_profile")),
        "default_profile": optional_bool(profile.get("default_profile")),
        "default_profile_image": optional_bool(profile.get("default_profile_image")),
        "description": clean_text(profile.get("description")),
        "location": clean_text(profile.get("location")),
        "url": clean_text(profile.get("url")),
        "followers_count": nonnegative(profile.get("followers_count")),
        "friends_count": nonnegative(profile.get("friends_count")),
        "listed_count": nonnegative(profile.get("listed_count")),
        "statuses_count": nonnegative(profile.get("statuses_count")),
        "favourites_count": nonnegative(profile.get("favourites_count")),
    }


def read_twibot20(root: Path) -> dict[str, list[Account]]:
    output = {"train": [], "dev": [], "test": []}
    for split in output:
        path = root / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        for record in iter_json_array(path):
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record in {path}")
            raw_tweets = record.get("tweet") if isinstance(record.get("tweet"), list) else []
            tweets = [text for item in raw_tweets if (text := clean_text(item)) is not None]
            output[split].append(
                Account(
                    str(record.get("ID", "")).strip(),
                    split,
                    canonical_label(record.get("label")),
                    twibot20_values(record),
                    tweets,
                )
            )
    return output


def read_csv_map(path: Path, column: str) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            account_id = normalize_account_id(row.get("id"))
            if account_id in output:
                raise ValueError(f"Duplicate account in {path}: {account_id}")
            output[account_id] = str(row.get(column, "")).strip()
    return output


def twibot22_values(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("public_metrics") if isinstance(record.get("public_metrics"), dict) else {}
    return {
        "name": clean_text(record.get("name")),
        "screen_name": clean_text(record.get("username")),
        "protected": optional_bool(record.get("protected")),
        "verified": optional_bool(record.get("verified")),
        "has_extended_profile": None,
        "default_profile": None,
        "default_profile_image": None,
        "description": clean_text(record.get("description")),
        "location": clean_text(record.get("location")),
        "url": clean_text(record.get("url")),
        "followers_count": nonnegative(metrics.get("followers_count")),
        "friends_count": nonnegative(metrics.get("following_count")),
        "listed_count": nonnegative(metrics.get("listed_count")),
        "statuses_count": nonnegative(metrics.get("tweet_count")),
        "favourites_count": None,
    }


def build_twibot22_index(root: Path, database: Path, tweet_limit: int) -> None:
    required = [root / "split.csv", root / "label.csv", root / "user.json"]
    required.extend(root / f"tweet_{index}.json" for index in range(9))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TwiBot-22 files: {missing}")
    splits = {key: canonical_split(value) for key, value in read_csv_map(root / "split.csv", "split").items()}
    labels = {key: canonical_label(value) for key, value in read_csv_map(root / "label.csv", "label").items()}
    if set(splits) != set(labels):
        raise ValueError("TwiBot-22 split and label account sets differ")
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS accounts (account_id TEXT PRIMARY KEY, split TEXT NOT NULL, label INTEGER NOT NULL, values_json TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS tweets (account_id TEXT NOT NULL, position INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY(account_id, position))")
    connection.execute("DELETE FROM accounts")
    connection.execute("DELETE FROM tweets")
    seen = set()
    for record in iter_json_array(root / "user.json"):
        if not isinstance(record, dict):
            continue
        account_id = normalize_account_id(record.get("id"))
        if account_id not in splits:
            continue
        connection.execute(
            "INSERT INTO accounts VALUES (?, ?, ?, ?)",
            (account_id, splits[account_id], labels[account_id], json.dumps(twibot22_values(record), ensure_ascii=False, separators=(",", ":"))),
        )
        seen.add(account_id)
    if seen != set(splits):
        raise ValueError(f"TwiBot-22 user coverage mismatch: {len(seen)} of {len(splits)}")
    connection.commit()
    counts: dict[str, int] = {}
    for part in range(9):
        for record in iter_json_array(root / f"tweet_{part}.json"):
            if not isinstance(record, dict):
                continue
            try:
                account_id = normalize_account_id(record.get("author_id"))
            except ValueError:
                continue
            if account_id not in splits:
                continue
            position = counts.get(account_id, 0)
            if position >= tweet_limit:
                continue
            counts[account_id] = position + 1
            text = clean_text(record.get("text"))
            if text is not None:
                connection.execute("INSERT INTO tweets VALUES (?, ?, ?)", (account_id, position, text))
        connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS tweets_account ON tweets(account_id, position)")
    connection.commit()
    connection.close()


def iter_twibot22(database: Path, split: str, batch_size: int) -> Iterator[list[Account]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    cursor = connection.execute("SELECT account_id, label, values_json FROM accounts WHERE split = ? ORDER BY rowid", (split,))
    while rows := cursor.fetchmany(batch_size):
        identifiers = [str(row["account_id"]) for row in rows]
        placeholders = ",".join("?" for _ in identifiers)
        tweets = {identifier: [] for identifier in identifiers}
        query = f"SELECT account_id, text FROM tweets WHERE account_id IN ({placeholders}) ORDER BY account_id, position"
        for tweet in connection.execute(query, identifiers):
            tweets[str(tweet[0])].append(str(tweet[1]))
        yield [
            Account(
                str(row["account_id"]),
                split,
                int(row["label"]),
                json.loads(str(row["values_json"])),
                tweets[str(row["account_id"])],
            )
            for row in rows
        ]
    connection.close()
