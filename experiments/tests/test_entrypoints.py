import ast
import csv
import json

import numpy as np

from experiments import worker
from experiments.config import ROOT, asset_names, notebook_path
from experiments.run import write_metric_table


def test_relocated_split_root_is_the_cohort_parent():
    from experiments.experiment_split import ROOT as split_root
    assert split_root == ROOT
    assert (split_root / "NACC").is_dir()


def test_nacc_primekg_switches_assets_and_result_label_together():
    source = '''ENTITY_EMB_NPY = "old.npy"
ENTITY2ID_FILE = "old.json"
PATIENT_QUERY_DICT = "old-patients.json"
result = run_ablation(dataset="NACC", graph="DementiaHKG")
'''
    tree = worker.adapt_source(source, "NACC", "PrimeKG")
    captured = {}
    namespace = {"run_ablation": lambda **kwargs: captured.update(kwargs)}
    exec(compile(tree, "test", "exec"), namespace)
    assert "NACC-PrimeKG-Embeddings.npy" in namespace["ENTITY_EMB_NPY"]
    assert "NACC-PrimeKG-Entities.json" in namespace["PATIENT_QUERY_DICT"]
    assert captured["graph"] == "PrimeKG"


def test_worker_stops_after_training_before_legacy_followup_cells(tmp_path, monkeypatch):
    path = tmp_path / "test.ipynb"
    cells = ["x = 1", "results = run_ablation()", "raise RuntimeError('legacy plotting must not run')"]
    path.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": [source]} for source in cells
    ]}), encoding="utf-8")
    monkeypatch.setattr(worker, "notebook_path", lambda *_: path)
    assert len(worker.training_cells("NACC", "PrimeKG")) == 2


def test_all_twelve_training_prefixes_compile_and_have_graph_assets():
    for dataset in ("ADNI", "AIBL", "NACC", "PPMI"):
        for graph in ("CustomKG", "PrimeKG", "DementiaHKG"):
            assert notebook_path(dataset, graph).is_file()
            cells = worker.training_cells(dataset, graph)
            for index, source in cells:
                compile(worker.adapt_source(source, dataset, graph), f"cell{index}", "exec")
            assert asset_names(dataset, graph)


def test_metric_table_reports_training_mean_and_sample_sd(tmp_path):
    directory = tmp_path / "NACC" / "DementiaHKG"
    directory.mkdir(parents=True)
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "accuracy", "precision", "recall", "f1", "auc"))
        writer.writeheader()
        for seed, score in zip((11, 23, 37, 51, 67), (0.5, 0.6, 0.7, 0.8, 0.9)):
            writer.writerow({"seed": seed, **dict.fromkeys(("accuracy", "precision", "recall", "f1", "auc"), score)})
    write_metric_table(tmp_path, "NACC")
    with (tmp_path / "NACC" / "metrics_mean_sd.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert np.isclose(float(row["f1_mean"]), 0.7)
    assert np.isclose(float(row["f1_sd"]), np.sqrt(0.025))
