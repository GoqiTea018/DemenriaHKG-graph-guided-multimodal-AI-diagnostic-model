import csv
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.kg_ablation import run_ablation, sample_identity
from experiments.kg_ablation_stats import aligned_arrays, main, paired_comparison


def test_identity_keeps_visits_distinct_and_patient_together():
    frames = [pd.DataFrame({"filename": ["visit_1", "visit_2"], "RID": [7, 7]})]
    records, patients = sample_identity(frames, ["AD"])
    assert len(set(records)) == 2
    assert list(patients) == ["7", "7"]


def test_identity_extracts_subject_from_cohort_scan_names():
    adni = pd.DataFrame({"filename": ["ADNI_002_S_0413_MR_a.nii", "ADNI_002_S_0413_MR_b.nii"]})
    _, patients = sample_identity([adni], ["AD"], dataset="ADNI")
    assert list(patients) == ["ADNI_002_S_0413"] * 2
    aibl = pd.DataFrame({"filename": ["AIBL_42_MR_a.nii", "AIBL_42_MR_b.nii"]})
    _, patients = sample_identity([aibl], ["AD"], dataset="AIBL")
    assert list(patients) == ["AIBL_42"] * 2


def test_test_set_is_evaluated_once_after_validation_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_ABLATION_OUTPUT", str(tmp_path))
    x = torch.eye(3).repeat(3, 1)
    y = torch.tensor([0, 1, 2] * 3)
    kg = torch.zeros(9, 1, 1)
    mask = torch.zeros(9, 1, dtype=torch.bool)
    loader = DataLoader(TensorDataset(x, kg, mask, y), batch_size=3)
    test_loader = DataLoader(TensorDataset(x, kg, mask, y), batch_size=3)
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, features, _kg, _mask):
            calls.append("predict")
            return self.linear(features)

    validations = iter([3.0, 1.0, 2.0] * 2)

    def train_epoch(*_args):
        return 1.0, 0.5, 0.5, 0.5, 0.5

    def validate_epoch(_model, seen_loader, *_args):
        assert seen_loader is loader
        return next(validations), 0.5, 0.5, 0.5, 0.5

    results, _ = run_ablation(
        dataset="ADNI", graph="CustomKG", make_model=Model,
        train_loader=loader, val_loader=loader, test_loader=test_loader,
        train_epoch=train_epoch, validate_epoch=validate_epoch,
        epochs=3, learning_rate=1e-3, weight_decay=0, device="cpu",
        test_record_ids=[f"r{i}" for i in range(9)],
        test_patient_ids=[f"p{i}" for i in range(9)], seeds=(11, 23),
    )
    assert [run["best_epoch"] for run in results] == [2, 2]
    assert len(calls) == 6  # Three test batches for each of two seeds.
    with (tmp_path / "ADNI" / "CustomKG" / "seed_11.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert rows[0]["record_id"] == "r0"


def test_paired_comparison_clusters_repeated_patient_records():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    units = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p4", "p5"])
    proposed = np.stack([np.eye(3)[y] * 0.8 + 0.2 / 3] * 5)
    baseline = np.stack([np.roll(np.eye(3)[y], 1, axis=1) * 0.7 + 0.3 / 3] * 5)
    result = paired_comparison(y, units, proposed, baseline, "f1", "ovr",
                               100, np.random.default_rng(1))
    assert result["units"] == 6
    assert result["difference"] > 0
    assert 0 < result["p_value"] <= 1


def test_statistical_comparison_rejects_unpaired_records():
    left = {seed: {"record_a": {"record_id": "record_a", "patient_id": "p1",
                                "y_true": "0", "prob_0": "0.8", "prob_1": "0.2"}}
            for seed in (11, 23, 37, 51, 67)}
    right = {seed: {"record_b": {"record_id": "record_b", "patient_id": "p1",
                                 "y_true": "0", "prob_0": "0.7", "prob_1": "0.3"}}
             for seed in (11, 23, 37, 51, 67)}
    with pytest.raises(ValueError, match="same test records"):
        aligned_arrays(left, right, "patient")


@pytest.mark.parametrize("dataset", ["ADNI", "NACC"])
def test_comparison_cli_writes_two_holm_adjusted_results(tmp_path, monkeypatch, dataset):
    y = [0, 1, 2] * 4
    for graph in ("DementiaHKG", "CustomKG", "PrimeKG"):
        directory = tmp_path / dataset / graph
        directory.mkdir(parents=True)
        for seed in (11, 23, 37, 51, 67):
            with (directory / f"seed_{seed}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("dataset", "graph", "seed", "record_id", "patient_id",
                                 "y_true", "y_pred", "prob_0", "prob_1", "prob_2"))
                for i, label in enumerate(y):
                    predicted = label if graph == "DementiaHKG" else (label + i % 2) % 3
                    probabilities = [0.05] * 3
                    probabilities[predicted] = 0.9
                    writer.writerow((dataset, graph, seed, f"r{i}", f"p{i}",
                                     label, predicted, *probabilities))
    monkeypatch.setattr(sys, "argv", ["kg_ablation_stats.py", str(tmp_path), dataset,
                                      "--metric", "f1", "--resamples", "100"])
    main()
    with (tmp_path / dataset / "paired_f1_patient.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(float(row["holm_p"]) >= float(row["p_value"]) for row in rows)
