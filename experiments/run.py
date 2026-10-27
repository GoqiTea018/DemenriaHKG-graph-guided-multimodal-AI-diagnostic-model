"""Check inputs, run three KG configurations, and compute paired statistics.

From the project folder:
    python -m experiments.run --dataset NACC --stage check
    python -m experiments.run --dataset NACC --stage all
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from .config import DATASETS, GRAPHS, MRI, ROOT, SEEDS, asset_names, notebook_path, resolve_graph_asset


def check_inputs(dataset, graphs):
    """Read metadata and check paths, without loading MRI volumes or training."""
    errors = []
    for module in ("numpy", "pandas", "sklearn", "torch", "nibabel", "matplotlib", "tqdm"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing Python dependency: {module}")
    for csv_name, folder in MRI[dataset]:
        path, directory = ROOT / dataset / csv_name, ROOT / dataset / folder
        if not path.is_file():
            errors.append(f"Missing cohort CSV: {path}")
            continue
        if not directory.is_dir():
            errors.append(f"MRI data not present: {directory}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        # These notebooks extract MRI for every CSV row before the PD slice.
        name_column = "filename" if "filename" in fields else fields[1]
        missing = sum(not (directory / row[name_column]).is_file() for row in rows)
        if missing:
            errors.append(f"{directory}: {missing}/{len(rows)} MRI files not present")
        else:
            print(f"OK {dataset}/{folder}: {len(rows)} MRI paths")
    from .worker import training_cells
    for graph in graphs:
        if not notebook_path(dataset, graph).is_file():
            errors.append(f"Missing notebook: {notebook_path(dataset, graph)}")
            continue
        training_cells(dataset, graph)  # Validate the training prefix's syntax.
        for filename in asset_names(dataset, graph).values():
            path = Path(resolve_graph_asset(dataset, graph, filename))
            if not path.is_file():
                errors.append(f"Missing {graph} asset: {path}")
    for error in errors:
        print(f"MISSING {error}")
    if not errors:
        print(f"OK {dataset}: input files and training entry points are ready")
    return errors


def write_metric_table(output, dataset):
    """Aggregate the five-run summaries without pooling test predictions."""
    import numpy as np
    rows = []
    for graph in GRAPHS:
        path = output / dataset / graph / "summary.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            runs = list(csv.DictReader(handle))
        if len(runs) != len(SEEDS) or {int(row["seed"]) for row in runs} != set(SEEDS):
            raise ValueError(f"Incomplete five-run summary: {path}")
        row = {"dataset": dataset, "graph": graph, "runs": len(runs)}
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            values = [float(run[metric]) for run in runs]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    if rows:
        destination = output / dataset / "metrics_mean_sd.csv"
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {destination}")


def run_process(arguments, env):
    subprocess.run([sys.executable, "-u", "-m", *arguments], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True,
                        help="Choose exactly one dataset for this invocation")
    parser.add_argument("--stage", choices=("check", "train", "stats", "all"), default="check")
    parser.add_argument("--graphs", nargs="+", choices=GRAPHS, default=list(GRAPHS),
                        help="KG variants to train/check; statistics compare all three")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--metrics", nargs="+", choices=("f1", "accuracy", "auc", "precision", "recall"), default=["f1"])
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("Use at least 100 resamples")
    datasets = (args.dataset,)
    output = args.output.resolve()
    if args.stage in ("check", "train", "all"):
        failed = False
        for dataset in datasets:
            failed = bool(check_inputs(dataset, args.graphs)) or failed
        if failed:
            print("Place the listed inputs at the configured paths, then rerun. No training started.")
            return 2
    if args.stage == "check":
        print("Check only: no models trained and no p-values computed.")
        return 0
    env = os.environ.copy()
    env["KG_ABLATION_OUTPUT"] = str(output)
    env["MPLBACKEND"] = "Agg"
    for dataset in datasets:
        if args.stage in ("train", "all"):
            for graph in args.graphs:
                print(f"Training {dataset}/{graph}: five seeds {SEEDS}", flush=True)
                run_process(["experiments.worker", dataset, graph], env)
            write_metric_table(output, dataset)
        if args.stage in ("stats", "all"):
            for metric in args.metrics:
                run_process(["experiments.kg_ablation_stats", str(output), dataset,
                             "--metric", metric, "--resamples", str(args.resamples)], env)
            write_metric_table(output, dataset)
    print(f"Finished. Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
