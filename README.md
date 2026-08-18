# FIESL

This repository contains the experiments of Fine-grained Internal Evidence Structure Learning (FIESL). FIESL performs account-local social bot detection without reading account-to-account edges during training or inference.

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
  map_raw_fields_to_units.py
  prepare_data.py
  run_pipeline.py
  run_five_seeds.py
  validate_release.py
src/fiesl/
  data.py
  encoding.py
  features.py
  metrics.py
  model.py
  prepare.py
  raw.py
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

No processed tensors or model weights are required from the authors. The repository builds the interaction-free representation from the official source files. Its exact tensor contract and text-encoding policy are documented in `docs/DATA.md`.

## Complete pipeline

The default commands download `FacebookAI/roberta-base` from [Hugging Face](https://huggingface.co/FacebookAI/roberta-base), encode the official raw data, fit preprocessing on Train only, create the nine-slot representation, and train the published default seed:

```bash
python scripts/run_pipeline.py \
  --config configs/fiesl_twibot20.json \
  --raw-root data/TwiBot-20 \
  --hf-cache-dir models/huggingface

python scripts/run_pipeline.py \
  --config configs/fiesl_twibot22.json \
  --raw-root data/TwiBot-22 \
  --hf-cache-dir models/huggingface
```

Add `--five-seeds` to run all five paper seeds. For an offline model copy, replace `--hf-cache-dir models/huggingface` with `--model-path /absolute/path/to/roberta-base`. Model files stay outside version control.

Preparation can also be run separately:

```bash
python scripts/prepare_data.py --config configs/fiesl_twibot20.json --raw-root data/TwiBot-20
python scripts/check_data.py --config configs/fiesl_twibot20.json
```

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

Each completed epoch records Train, Dev, and observation-only Test metrics. 

## LLM-assisted unit mapping

`scripts/map_raw_fields_to_units.py` is a standalone provenance and replay script. It contains the TwiBot-20 and TwiBot-22 field-name schemas, the constrained LLM prompt, the frozen nine-unit response, output validation, and the resulting raw-field-to-unit mapping. It receives schema metadata only and never reads dataset records, labels, graph files, metrics, or training outputs.

The script is not part of the training call path. The published training implementation directly uses the frozen 9-by-10 membership matrix in `src/fiesl/model.py`; it does not call an LLM or require an API key.

```bash
python scripts/map_raw_fields_to_units.py --print-prompt
python scripts/map_raw_fields_to_units.py --output frozen_unit_mapping.json
python scripts/map_raw_fields_to_units.py --response-file candidate.json --output candidate_mapping.json
```

## Baselines

Third-party baseline source code is not redistributed. `docs/BASELINES.md` is the single baseline reference file and records upstream sources, TwiBot-20 and TwiBot-22 settings, the five-seed protocol, and compatibility rules.

## Validation

```bash
python scripts/validate_release.py
pytest -q
```
