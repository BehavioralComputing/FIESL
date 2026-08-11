# Reproducibility protocol

## Fixed seeds

All main experiments use the five seeds in `configs/seeds.json`:

`1503191042`, `1284006632`, `2049683099`, `375449128`, and `545955441`.

## Model and optimization

FIESL uses hidden, encoder-inner, pair, and aggregation dimensions of 128, 256, 128, and 256. The batch size is 128, the learning rate is `3e-4`, weight decay is `1e-4`, gradient clipping is 5.0, and training runs for at most 55 epochs.

## Selection and evaluation

Train is used only for fitting. Dev Bot-F1 is the sole checkpoint and early-stopping selector. A Test observation is persisted after every completed epoch, but Test is never used to select a checkpoint, hyperparameter, threshold, or model. The reported result is the Test observation aligned with the Dev-selected epoch.

Every run writes:

- `run_manifest.json`
- `status.json`
- `history.jsonl`
- `checkpoint_best_dev.pt`
- `metrics.json`

The five-seed paper result is the arithmetic mean and sample standard deviation over the five completed run directories. Do not replace failed seeds or select seeds using Test performance.
