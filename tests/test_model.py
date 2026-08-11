import torch

from fiesl.data import EvidenceBatch
from fiesl.model import FIESL


INPUT_DIMS = [783, 773, 1, 4, 4, 4, 768, 9, 15]


def make_batch() -> EvidenceBatch:
    count = 4
    typed = torch.zeros((count, 9, 783), dtype=torch.float32)
    available = torch.ones((count, 9), dtype=torch.bool)
    available[:, 2] = False
    quality = torch.zeros((count, 9, 8), dtype=torch.float32)
    for slot, width in enumerate(INPUT_DIMS):
        if slot != 2:
            typed[:, slot, :width] = torch.randn(count, width)
            quality[:, slot] = torch.randn(count, 8)
    return EvidenceBatch(
        account_ids=[str(index) for index in range(count)],
        labels=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        typed_inputs=typed,
        input_dims=torch.tensor(INPUT_DIMS, dtype=torch.int64),
        availability_mask=available,
        quality_features=quality,
    ).validate()


def test_forward_shape() -> None:
    model = FIESL(INPUT_DIMS, 8)
    logits = model(make_batch())
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_relation_contract() -> None:
    model = FIESL(INPUT_DIMS, 8)
    assert model.pair_left.numel() == 36
    assert torch.equal(model.incidence.sum(dim=0), torch.full((36,), 2))
    assert torch.equal(model.incidence.sum(dim=1), torch.full((9,), 8))

