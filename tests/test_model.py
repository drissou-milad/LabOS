import torch

from src.model import CellCNN


def test_output_shape():
    model = CellCNN()
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 1)


def test_returns_logits_not_probabilities():
    """Regression test for the original bug: the model must NOT end in a
    Sigmoid. If someone re-adds one, outputs will be constrained to [0, 1]
    for every input — this test would start failing."""
    model = CellCNN()
    x = torch.randn(64, 1, 32, 32) * 50  # deliberately large-scale input
    out = model(x)
    assert out.min() < 0 or out.max() > 1, (
        "Output looks like it's been squashed into [0, 1] — did a Sigmoid "
        "get added back to CellCNN.classifier?"
    )


def test_gradients_flow():
    """Regression test for the original bug: a forward+backward pass must
    produce non-zero gradients on a batch with both classes present."""
    model = CellCNN()
    x = torch.randn(8, 1, 32, 32)
    y = torch.tensor([0.0, 1.0] * 4)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(model(x).squeeze(1), y)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert max(grad_norms) > 0.0


def test_accepts_batch_size_one():
    model = CellCNN()
    x = torch.randn(1, 1, 32, 32)
    model.eval()
    out = model(x)
    assert out.shape == (1, 1)
