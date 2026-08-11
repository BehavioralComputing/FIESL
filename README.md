# FIESL

This repository contains the minimal implementation used for the main experiments of Fine-grained Internal Evidence Structure Learning (FIESL). FIESL performs account-local social bot detection without reading account-to-account edges during training or inference.

## Repository layout

```text
configs/
  fiesl_twibot20.json
  fiesl_twibot22.json
  seeds.json
data/
  TwiBot-20/
  TwiBot-22/
  processed/
docs/
  BASELINES.md
  DATA.md
  REPRODUCIBILITY.md
scripts/
  check_data.py
  run_five_seeds.py
  validate_release.py
src/fiesl/
  data.py
  metrics.py
  model.py
  training.py
tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

## Data

Dataset files are intentionally excluded. Apply for access through the official [TwiBot-20 repository](https://github.com/BunsenFeng/TwiBot-20) and the official [TwiBot-22 repository](https://github.com/LuoUndergradXJTU/TwiBot-22). Put downloaded files under `data/TwiBot-20` and `data/TwiBot-22`, respectively. The TwiBot-22 download is distributed through the [official Google Drive folder](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs?usp=sharing).

The trainer reads interaction-free representation files from `data/processed/<dataset>/{train,dev,test}.pt`. Their exact tensor contract and text-encoding policy are documented in `docs/DATA.md`.

## Main experiments

Run one seed:

```bash
python -m fiesl.training --config configs/fiesl_twibot20.json
python -m fiesl.training --config configs/fiesl_twibot22.json
```

The commands above use each dataset configuration's published `default_seed`.
Pass `--seed <value>` to run another seed from `configs/seeds.json`.

Run the five paper seeds sequentially:

```bash
python scripts/run_five_seeds.py --config configs/fiesl_twibot20.json
python scripts/run_five_seeds.py --config configs/fiesl_twibot22.json
```

Each completed epoch records Train, Dev, and observation-only Test metrics. Dev Bot-F1 is the sole checkpoint selector. Test observations are never used for model, hyperparameter, threshold, or checkpoint selection.

## Baselines

Third-party baseline source code is not redistributed. `docs/BASELINES.md` is the single baseline reference file and records upstream sources, TwiBot-20 and TwiBot-22 settings, the five-seed protocol, and compatibility rules.

## Validation

```bash
python scripts/validate_release.py
pytest -q
```
