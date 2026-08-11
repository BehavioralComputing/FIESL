# Data and representation contract

## Official datasets

| Dataset | Official source | Local raw directory |
|---|---|---|
| TwiBot-20 | https://github.com/BunsenFeng/TwiBot-20 | `data/TwiBot-20` |
| TwiBot-22 | https://github.com/LuoUndergradXJTU/TwiBot-22 | `data/TwiBot-22` |

TwiBot-22 access is managed through the Google Drive folder linked by its official repository. Dataset licenses and platform redistribution rules apply. Raw data, derived text, account identifiers, embeddings, and labels are excluded from this release.

## Information boundary

FIESL uses only focal-account evidence. Representation generation must not read `edge.csv`, neighboring accounts, support labels, or any feature derived from account-to-account topology.

## Official text encoding

Both datasets use [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) at pinned revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, maximum token length 128, masked token mean pooling, L2 normalization, and account-level mean pooling over own-tweet vectors. The model is loaded either from Hugging Face or from a complete local model directory passed with `--model-path`. No weights are included in the repository.

- TwiBot-20 uses all available own tweets.
- TwiBot-22 uses the first 20 own tweets in official dataset order and only official supported text fields.

Numeric and linguistic-style statistics must be fitted on the official Train partition only and then frozen for Dev and Test.

## Required official files

TwiBot-20 requires `train.json`, `dev.json`, and `test.json`. Each record supplies `ID`, `label`, `profile`, and `tweet` as defined by the official release. `support.json` is not read.

TwiBot-22 requires `split.csv`, `label.csv`, `user.json`, and `tweet_0.json` through `tweet_8.json`. The preparation code joins tweets only through their documented `author_id`. It never opens `edge.csv`, `hashtag.json`, or `list.json`. A local SQLite staging index is created under `data/processed/TwiBot-22` so the 1M-account preparation remains bounded in memory.

## Evidence order

The nine slots are fixed in this order:

1. Identity
2. Profile
3. Account Maturity
4. Popularity
5. Social Ratio
6. Activity Intensity
7. Content Semantics
8. Content Diversity
9. Linguistic Style

Account Maturity is inactive when no exact observation date is available. Any other unavailable or unsupported evidence remains in its fixed slot with a false availability mask and a zero vector.

## Processed files

Each dataset directory under `data/processed` contains `train.pt`, `dev.pt`, and `test.pt`. Every file is a dictionary saved with `torch.save` and contains:

| Key | Type | Shape |
|---|---|---|
| `account_ids` | list of strings | `N` |
| `labels` | int64 tensor | `[N]` |
| `typed_inputs` | float32 tensor | `[N, 9, 783]` |
| `input_dims` | int64 tensor | `[9]` |
| `availability_mask` | bool tensor | `[N, 9]` |
| `quality_features` | float32 tensor | `[N, 9, 8]` |

The fixed input dimensions are `[783, 773, 1, 4, 4, 4, 768, 9, 15]`. Values outside each slot's declared width must be zero. Values and quality features for unavailable slots must also be zero.

Run `python scripts/check_data.py --config configs/fiesl_twibot20.json` or the TwiBot-22 equivalent before training.

Each processed dataset also contains `preparation_manifest.json`. It records the encoder source and resolved revision, the text contract, Train-only preprocessing statistics, official split counts, source-file sizes, the zero-topology access list, and a canonical contract hash. Add `--hash-source-files` during preparation when full raw-file SHA-256 values are required.
