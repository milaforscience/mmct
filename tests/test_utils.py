from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils import composition
from utils.get_compute_device import get_compute_device
from utils.optim import SWA
from utils.utils import EDMDataset, DummyScaler, Scaler, linear


def test_parse_formula_expands_nested_group_and_metallofullerene_marker():
    assert dict(composition.parse_formula("Li3Fe2(PO4)3")) == {
        "Li": 3.0,
        "Fe": 2.0,
        "P": 3.0,
        "O": 12.0,
    }
    assert dict(composition.parse_formula("Y3N@C80")) == {
        "Y": 3.0,
        "N": 1.0,
        "C": 80.0,
    }


def test_parse_formula_rejects_unconsumed_characters():
    with pytest.raises(composition.CompositionError):
        composition.parse_formula("Fe2O3?")


def test_element_property_data_root_is_relative_to_source_file():
    expected_root = Path(composition.__file__).resolve().parents[1]

    assert Path(composition.dirpath) == expected_root


def test_generate_features_builds_element_property_features(tmp_path, monkeypatch):
    properties = tmp_path / "data" / "element_properties"
    properties.mkdir(parents=True)
    pd.DataFrame(
        {"element": ["H", "O"], "property": [1.0, 3.0]}
    ).to_csv(properties / "test.csv", index=False)
    monkeypatch.setattr(composition, "dirpath", str(tmp_path))
    data = pd.DataFrame({"formula": ["H2O"], "target": [4.5]})

    features, targets, formulas, skipped = composition.generate_features(
        data, elem_prop="test"
    )

    assert features.shape == (1, 6)
    assert targets.tolist() == [4.5]
    assert formulas.tolist() == ["H2O"]
    assert skipped == []


def test_get_compute_device_can_be_forced_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert get_compute_device(force_cpu=True) == torch.device("cpu")


def test_linear_applies_matrix_transform_and_bias():
    inputs = torch.tensor([[1.0, 2.0]])
    weights = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    bias = torch.tensor([1.0, -1.0])

    assert torch.equal(linear(inputs, weights, bias), torch.tensor([[12.0, 16.0]]))


def test_scalers_round_trip_and_dummy_scaler_preserves_values():
    values = torch.tensor([1.0, 2.0, 3.0])
    scaler = Scaler(values)
    scaled = scaler.scale(values)

    assert torch.allclose(scaler.unscale(scaled), values)
    assert torch.equal(DummyScaler(values).scale(values), values)


def test_edm_dataset_returns_tensors_and_formula():
    dataset = EDMDataset(
        (np.array([[[1.0], [0.5]]]), np.array([2.0]), np.array(["H2"])),
        n_comp=1,
    )

    features, target, formula = dataset[0]

    assert features.shape == (2, 1)
    assert target.item() == 2.0
    assert formula == "H2"


def test_swa_can_swap_to_collected_average():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = SWA(torch.optim.SGD([parameter], lr=0.1))
    group = optimizer.param_groups[0]
    optimizer.update_swa_group(group)
    parameter.data.fill_(4.0)
    optimizer.update_swa_group(group)
    averaged_value = optimizer.state[parameter]["swa_buffer"].clone()

    optimizer.swap_swa_sgd()

    assert torch.equal(parameter.data, averaged_value)
