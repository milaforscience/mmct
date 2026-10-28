from pathlib import Path

import torch
from torch import nn

from models import kingcrab
from models.kingcrab import CrabNet, FractionalEncoder, ResidualNetwork
from models.model import Model


def test_residual_network_returns_batch_by_output_tensor():
    network = ResidualNetwork(input_dim=4, output_dim=2, hidden_layer_dims=[4, 3])

    output = network(torch.randn(5, 4))

    assert output.shape == (5, 2)


def test_fractional_encoder_clamps_values_and_preserves_batch_shape():
    encoder = FractionalEncoder(d_model=8, resolution=16)
    fractions = torch.tensor([[0.0, 0.5, 1.2]])

    encoded = encoder(fractions)

    assert encoded.shape == (1, 3, 4)
    assert torch.isfinite(encoded).all()


def test_crabnet_forward_handles_padding(monkeypatch):
    element_vectors = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32
    ).numpy()
    csv_paths = []

    def fake_read_csv(path, *args, **kwargs):
        csv_paths.append(path)
        return kingcrab.pd.DataFrame(element_vectors, index=[1, 2, 3])

    monkeypatch.setattr(kingcrab.pd, "read_csv", fake_read_csv)
    model = CrabNet(out_dims=2, d_model=8, N=1, heads=2)
    src = torch.tensor([[1, 2, 0], [3, 0, 0]])
    fractions = torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]])

    output = model(src, fractions)

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    assert csv_paths == [
        Path(kingcrab.__file__).resolve().parents[1]
        / "data"
        / "element_properties"
        / "mat2vec.csv"
    ]


def test_model_uses_model_compute_device_without_reselecting_device():
    class StubModel(nn.Module):
        out_dims = 2
        d_model = 8
        N = 1
        heads = 2
        compute_device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    wrapped = Model(StubModel(), verbose=False)

    assert wrapped.compute_device == torch.device("cpu")
