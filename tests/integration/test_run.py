import json

import pytest

import run as run_module


def test_run_experiment_dispatches_each_fold_and_writes_test_summary(
    tmp_path, monkeypatch
):
    config = {
        "n_splits": 2,
        "test": True,
        "task": {
            "root": str(tmp_path / "benchmark"),
            "src": "$root",
            "target": "formation_energy",
        },
        "model": {},
    }
    model_calls = []
    result_calls = []
    model_sentinel = object()
    test_maes = {0: 0.2, 1: 0.4}

    def fake_get_model(task, fold, cfg, verbose):
        model_calls.append((task, fold, cfg["task"]["data_path"], cfg["model"]["name"]))
        return model_sentinel

    def fake_save_results(task, fold, cfg, file_name, verbose):
        result_calls.append((task, fold, file_name, verbose))
        if file_name.startswith("test"):
            return model_sentinel, test_maes[fold]
        return model_sentinel, 0.1

    monkeypatch.setattr(run_module, "get_model", fake_get_model)
    monkeypatch.setattr(run_module, "save_results", fake_save_results)
    monkeypatch.setattr(run_module, "__file__", str(tmp_path / "run.py"))

    run_module.run_experiment(config, model="crabnet", task="formation_energy")

    assert model_calls == [
        ("formation_energy", 0, str(tmp_path / "benchmark"), "crabnet"),
        ("formation_energy", 1, str(tmp_path / "benchmark"), "crabnet"),
    ]
    assert result_calls == [
        ("formation_energy", 0, "test0.csv", False),
        ("formation_energy", 0, "val0.csv", False),
        ("formation_energy", 1, "test1.csv", False),
        ("formation_energy", 1, "val1.csv", False),
    ]

    summary_path = tmp_path / "benchmark_test.txt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mean_text, std_text = summary["formation_energy"].split(" ± ")
    assert float(mean_text) == pytest.approx(0.3)
    assert float(std_text) == pytest.approx(0.1)
