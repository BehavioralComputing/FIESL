from __future__ import annotations

import torch


def binary_metrics(labels: torch.Tensor, probabilities: torch.Tensor, threshold: float) -> dict[str, float]:
    labels = labels.to(torch.int64).cpu()
    predictions = (probabilities.cpu() >= threshold).to(torch.int64)
    values: dict[str, float] = {}
    class_f1 = []
    for target in (0, 1):
        true_positive = int(((predictions == target) & (labels == target)).sum())
        false_positive = int(((predictions == target) & (labels != target)).sum())
        false_negative = int(((predictions != target) & (labels == target)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        class_f1.append(f1)
        if target == 1:
            values["precision"] = precision
            values["recall"] = recall
            values["f1"] = f1
    values["accuracy"] = float((predictions == labels).float().mean())
    values["macro_f1"] = sum(class_f1) / 2
    return values

