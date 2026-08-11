import torch

from fiesl.metrics import binary_metrics


def test_binary_metrics() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([0.1, 0.8, 0.7, 0.9])
    result = binary_metrics(labels, probabilities, 0.5)
    assert result["accuracy"] == 0.75
    assert 0 <= result["macro_f1"] <= 1
    assert 0 <= result["f1"] <= 1
