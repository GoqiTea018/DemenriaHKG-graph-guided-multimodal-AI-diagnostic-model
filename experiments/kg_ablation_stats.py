"""Paired, patient-clustered KG ablation comparisons from exported predictions.

Example: python -m experiments.kg_ablation_stats D:/private/kg_results ADNI --metric f1
The five seeds are averaged within each resample. All records belonging to a
patient, across all five seeds, move together. A record-level analysis is
possible only with an explicit --unit record choice.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import SEEDS


def read_predictions(root: Path, dataset: str, graph: str):
    runs = {}
    for seed in SEEDS:
        path = root / dataset / graph / f"seed_{seed}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(int(row["seed"]) != seed for row in rows):
            raise ValueError(f"Missing rows or incorrect seed in {path}")
        if any(row["dataset"] != dataset or row["graph"] != graph for row in rows):
            raise ValueError(f"Dataset/graph mismatch in {path}")
        runs[seed] = {row["record_id"]: row for row in rows}
        if len(runs[seed]) != len(rows):
            raise ValueError(f"Duplicate record IDs in {path}")
    return runs


def aligned_arrays(reference, candidate, unit):
    records = sorted(reference[SEEDS[0]])
    if not records:
        raise ValueError("No test records")
    truth, units, ref_probs, cand_probs = [], [], [], []
    for seed in SEEDS:
        if set(reference[seed]) != set(records) or set(candidate[seed]) != set(records):
            raise ValueError(f"The two graphs do not share the same test records for seed {seed}")
        ref_rows = [reference[seed][record] for record in records]
        cand_rows = [candidate[seed][record] for record in records]
        y = np.asarray([int(row["y_true"]) for row in ref_rows])
        if not np.array_equal(y, [int(row["y_true"]) for row in cand_rows]):
            raise ValueError(f"Label mismatch between graphs for seed {seed}")
        if truth and not np.array_equal(truth[0], y):
            raise ValueError("Test labels changed between seeds")
        ids = [row["patient_id"] if unit == "patient" else row["record_id"] for row in ref_rows]
        other_ids = [row["patient_id"] if unit == "patient" else row["record_id"] for row in cand_rows]
        if ids != other_ids or any(not value for value in ids):
            raise ValueError("Patient IDs are missing or disagree; provide them or explicitly choose --unit record")
        if units and units[0] != ids:
            raise ValueError("Patient IDs changed between seeds")
        prob_cols = sorted((key for key in ref_rows[0] if key.startswith("prob_")),
                           key=lambda key: int(key.split("_")[1]))
        if not prob_cols or any(set(prob_cols) != {key for key in row if key.startswith("prob_")}
                                for row in ref_rows + cand_rows):
            raise ValueError("Missing or inconsistent probability columns")
        truth.append(y)
        units.append(ids)
        ref_probs.append(np.asarray([[float(row[key]) for key in prob_cols] for row in ref_rows]))
        cand_probs.append(np.asarray([[float(row[key]) for key in prob_cols] for row in cand_rows]))
    return truth[0], np.asarray(units[0]), np.asarray(ref_probs), np.asarray(cand_probs)


def paired_comparison(y, units, proposed, baseline, metric, auc_mode, resamples, rng):
    if len(y) != proposed.shape[1] or proposed.shape != baseline.shape:
        raise ValueError("Prediction shapes or labels do not align")
    groups = [np.flatnonzero(units == value) for value in np.unique(units)]
    if len(groups) < 2:
        raise ValueError("At least two independent sampling units are needed")

    def score(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
        if metric == "accuracy":
            return accuracy_score(labels, predictions)
        if metric == "precision":
            return precision_score(labels, predictions, average="macro", zero_division=0)
        if metric == "recall":
            return recall_score(labels, predictions, average="macro", zero_division=0)
        if metric == "f1":
            return f1_score(labels, predictions, average="macro", zero_division=0)
        return roc_auc_score(labels, probabilities, multi_class=auc_mode, average="macro")

    def difference(indices, left, right):
        return np.mean([
            score(y[indices], left[seed, indices])
            - score(y[indices], right[seed, indices])
            for seed in range(len(SEEDS))
        ])

    all_indices = np.arange(len(y))
    observed = difference(all_indices, proposed, baseline)
    bootstrap = []
    attempts = 0
    while len(bootstrap) < resamples and attempts < resamples * 10:
        attempts += 1
        chosen = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        try:
            bootstrap.append(difference(indices, proposed, baseline))
        except ValueError:  # A resample may omit a class required for multiclass AUC.
            continue
    if len(bootstrap) < resamples:
        raise ValueError("Too few valid bootstrap samples; check class and patient counts")

    extreme = 0
    for _ in range(resamples):
        swap_group = rng.integers(2, size=len(groups)).astype(bool)
        swap = np.zeros(len(y), dtype=bool)
        for group, selected in zip(groups, swap_group):
            if selected:
                swap[group] = True
        left, right = proposed.copy(), baseline.copy()
        left[:, swap], right[:, swap] = baseline[:, swap], proposed[:, swap]
        extreme += abs(difference(all_indices, left, right)) >= abs(observed) - 1e-12
    return {
        "difference": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "p_value": (extreme + 1) / (resamples + 1),
        "units": len(groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("dataset", choices=("ADNI", "AIBL", "NACC", "PPMI"))
    parser.add_argument("--metric", choices=("accuracy", "precision", "recall", "f1", "auc"), default="f1")
    parser.add_argument("--unit", choices=("patient", "record"), default="patient")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("Use at least 100 resamples")
    reference = read_predictions(args.output_root, args.dataset, "DementiaHKG")
    auc_mode = "ovo" if args.dataset == "AIBL" else "ovr"
    results = []
    for graph in ("CustomKG", "PrimeKG"):
        candidate = read_predictions(args.output_root, args.dataset, graph)
        arrays = aligned_arrays(reference, candidate, args.unit)
        result = paired_comparison(*arrays, args.metric, auc_mode,
                                   args.resamples, np.random.default_rng(args.seed))
        results.append({"dataset": args.dataset, "proposed": "DementiaHKG",
                        "baseline": graph, "metric": args.metric, "unit": args.unit, **result})
    ordered = sorted(range(len(results)), key=lambda index: results[index]["p_value"])
    for rank, index in enumerate(ordered):
        results[index]["holm_p"] = min(1.0, (len(results) - rank) * results[index]["p_value"])
    # Holm adjusted p-values must be nondecreasing in raw-p order.
    for previous, current in zip(ordered, ordered[1:]):
        results[current]["holm_p"] = max(results[current]["holm_p"], results[previous]["holm_p"])
    destination = args.output_root / args.dataset / f"paired_{args.metric}_{args.unit}.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(result)
    print(f"Saved aggregate comparison: {destination}")


if __name__ == "__main__":
    main()
