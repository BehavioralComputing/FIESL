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

Both datasets use `roberta-base`, maximum token length 128, masked token mean pooling, L2 normalization, and account-level mean pooling over own-tweet vectors.

- TwiBot-20 uses all available own tweets.
- TwiBot-22 uses the first 20 own tweets in official dataset order and only official supported text fields.

Numeric and linguistic-style statistics must be fitted on the official Train partition only and then frozen for Dev and Test.

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

