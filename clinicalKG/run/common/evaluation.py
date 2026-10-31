"""Full PrimeKG / DemRKG / CustomKG classification and paired inference.

This entry point consumes existing KG embeddings. It never retrains DistMult.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .paths import CLINICALKG_ROOT, PROJECT_ROOT, ANALYSIS_ROOT, dataset_dir, full_primekg_dir, asset_path, result_path
from .config import GRAPHS, SEEDS, EXPERIMENT_TAG, MRI, asset_names, notebook_path, model_path, auc_mode as dataset_auc_mode

DATASET_CHOICES = {1: "ADNI", 2: "NACC", 3: "AIBL", 4: "PPMI"}
DEFAULT_OUTPUT = ANALYSIS_ROOT
PRIMEKG_DEFINITION = (
    "Full original PrimeKG: all input edge records and all entity/relation types; "
    "no seed-neighborhood extraction, blacklist filtering, or top-k pruning. "
    "One shared DistMult embedding and entity index for all cohorts."
)
LABELS = {"CustomKG": "CustomKG", "PrimeKG": "PrimeKG (Full original graph)",
          "DementiaHKG": "DemRKG"}
EVALUATION_PROTOCOL = {"model_selection": "final_epoch", "evaluation_split": "val",
                       "split": "original record-level stratified 64/16/20; random_state=32,30"}


def default_output_root(dataset):
    """All cohorts retain the original result directory and filenames."""
    return DEFAULT_OUTPUT


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def inspect_full_primekg():
    """Validate the saved full-graph training state against the exported index/matrix."""
    import numpy as np
    folder = full_primekg_dir()
    files = {name: folder / ("PrimeKG-" + name) for name in
             ("Embeddings.npy", "Entity2ID.json", "Relation2ID.json", "TrainingState.json")}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Full PrimeKG asset: {path}")
    state = json.loads(files["TrainingState.json"].read_text(encoding="utf-8"))
    if (state.get("relations") != 30 or state.get("records", 0) < 1
            or len(state.get("graph_sha256", "")) != 64):
        raise ValueError("Full PrimeKG training state must contain the original graph fingerprint and 30 relations")
    entities = json.loads(files["Entity2ID.json"].read_text(encoding="utf-8"))
    relations = json.loads(files["Relation2ID.json"].read_text(encoding="utf-8"))
    embedding = np.load(files["Embeddings.npy"], mmap_mode="r")
    if (embedding.shape != (state["entities"], state["embed_dim"])
            or sorted(entities.values()) != list(range(embedding.shape[0]))
            or len(relations) != state["relations"]):
        raise ValueError("Full PrimeKG embedding/index/training-state dimensions do not agree")
    return {"scope": "full_original_graph", "definition": PRIMEKG_DEFINITION,
            "raw_graph_sha256": state["graph_sha256"], "records": state["records"],
            "entities": state["entities"], "relations": state["relations"],
            "embed_dim": state["embed_dim"], "completed_kge_epochs": state["completed_epochs"],
            "kge_global_step": state["global_step"],
            "asset_files": {key: str(path.resolve()) for key, path in files.items()},
            "embedding_sha256": _hash(files["Embeddings.npy"]),
            "entity_index_sha256": _hash(files["Entity2ID.json"])}


def check_inputs(dataset, graphs=GRAPHS):
    """Metadata/path checks only: no model training and no MRI volume loading."""
    from .classification_worker import training_cells
    errors = []
    for module in ("torch", "torchvision", "numpy", "pandas", "sklearn", "nibabel", "matplotlib", "tqdm"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing dependency: {module}")
    for filename, folder in MRI[dataset]:
        csv_path, directory = dataset_dir(dataset) / filename, dataset_dir(dataset) / folder
        if not csv_path.is_file():
            errors.append(f"Missing cohort CSV: {csv_path}")
            continue
        if not directory.is_dir():
            errors.append(f"Missing MRI directory: {directory}")
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = list(reader)
        # PPMI's original loaders use row[1]: its MRI filename column is unnamed.
        if dataset == "PPMI":
            filename_index = 1
        elif "filename" in header:
            filename_index = header.index("filename")
        else:
            errors.append(f"Missing filename column: {csv_path}")
            continue
        malformed = sum(len(row) <= filename_index or not row[filename_index].strip()
                        for row in rows)
        if malformed:
            errors.append(f"{csv_path}: {malformed} rows lack an MRI filename")
            continue
        missing = sum(not (directory / row[filename_index]).is_file() for row in rows)
        if missing:
            errors.append(f"{directory}: {missing}/{len(rows)} MRI files missing")
    for graph in graphs:
        if not notebook_path(dataset, graph).is_file():
            errors.append(f"Missing notebook: {notebook_path(dataset, graph)}")
            continue
        for index, source in training_cells(dataset, graph):
            compile(source, f"{dataset}/{graph}:cell{index}", "exec")
        for filename in asset_names(dataset, graph).values():
            path = Path(asset_path(dataset, graph, filename))
            if not path.is_file():
                errors.append(f"Missing {graph} asset: {path}")
    for error in errors:
        print(error)
    if not errors:
        print(f"{dataset}: cohort paths, notebooks and KG assets are ready.")
    return errors


def _graph_manifest(dataset, graph, full_kg):
    from .experiment_split import cohort_manifest
    split = cohort_manifest(dataset)[["_patient", "_label", "_split"]].to_csv(index=False)
    assets = {key: Path(asset_path(dataset, graph, filename))
              for key, filename in asset_names(dataset, graph).items()}
    manifest = {"dataset": dataset, "graph": graph, "label": LABELS[graph],
            "training_seeds": list(SEEDS), **EVALUATION_PROTOCOL,
            "split_sha256": hashlib.sha256(split.encode("utf-8")).hexdigest(),
            "auc_mode": dataset_auc_mode(dataset),
            "notebook_sha256": _hash(notebook_path(dataset, graph)),
            "kg_assets": {key: {"path": str(path.resolve()), "sha256": _hash(path)}
                          for key, path in assets.items()},
            "primekg": full_kg if graph == "PrimeKG" else None}
    if dataset in ("AIBL", "PPMI"):
        manifest["model_sha256"] = _hash(model_path(dataset, graph))
        manifest["kg_padding_mask"] = "bool: False=valid KG entity; True=padding; finite null fallback for empty sequences"
    return manifest


def _complete(folder, dataset, graph):
    paths = [result_path(folder, dataset, graph, "summary.csv")]
    paths += [result_path(folder, dataset, graph, name) for seed in SEEDS
              for name in (f"seed_{seed}.csv", f"history_seed_{seed}.csv", f"last_seed_{seed}.pt")]
    if not all(path.is_file() for path in paths):
        return False
    with paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == len(SEEDS) and {int(row["seed"]) for row in rows} == set(SEEDS)


def _stream_training_process(command, env):
    """Forward worker output into the calling notebook cell as each line arrives."""
    process = subprocess.Popen(command, cwd=CLINICALKG_ROOT.parent, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    finally:
        process.stdout.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _clear_previous_run(folder, dataset, graph):
    """Clear this graph's old outputs and shared statistics when rerunning."""
    names = ["summary.csv", "run_manifest.json"]
    names += [name for seed in SEEDS for name in
              (f"seed_{seed}.csv", f"history_seed_{seed}.csv", f"best_seed_{seed}.pt", f"last_seed_{seed}.pt")]
    for name in names:
        (result_path(folder, dataset, graph, name)).unlink(missing_ok=True)
    shared = ["metrics_mean_sd.csv", "paired_statistics.csv", "analysis_definition.json", "training_curves.png"]
    shared += [f"paired_{metric}_patient.csv" for metric in ("precision", "recall", "f1", "accuracy", "auc")]
    for name in shared:
        (result_path(folder, dataset, None, name)).unlink(missing_ok=True)


def _matches_saved_manifest(saved, current):
    """Accept a recorded directory migration while retaining training provenance."""
    if saved == current:
        return True
    migration = CLINICALKG_ROOT / "run" / "record" / "ALL_notebook_layout_migration_2026-10-31.json"
    if not migration.is_file():
        return False
    entries = json.loads(migration.read_text(encoding="utf-8"))
    match = any(entry["dataset"] == current["dataset"]
                and entry["graph"] == current["graph"]
                and entry["before_sha256"] == saved.get("notebook_sha256")
                and entry["after_sha256"] == current.get("notebook_sha256")
                for entry in entries)
    if not match:
        return False
    expected = dict(current, notebook_sha256=saved["notebook_sha256"])
    return saved == expected


def train_classifiers(dataset, output_root=None, graphs=GRAPHS, reuse_completed=True):
    full_kg = inspect_full_primekg() if "PrimeKG" in graphs else None
    output_root = Path(default_output_root(dataset) if output_root is None else output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    errors = check_inputs(dataset, graphs)
    if errors:
        raise FileNotFoundError("Input check failed; see the listed paths. No training started.")
    env = os.environ.copy()
    env["KG_ABLATION_OUTPUT"] = str(output_root)
    env["KG_ABLATION_SEEDS"] = ",".join(map(str, SEEDS))
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join((str(CLINICALKG_ROOT.parent),
                                        str(CLINICALKG_ROOT), env.get("PYTHONPATH", "")))
    for graph in graphs:
        folder = output_root
        manifest_path = result_path(folder, dataset, graph, "run_manifest.json")
        manifest = _graph_manifest(dataset, graph, full_kg)
        if not reuse_completed:
            _clear_previous_run(folder, dataset, graph)
        elif manifest_path.is_file():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not _matches_saved_manifest(old, manifest):
                raise ValueError(f"Results use different assets/settings: {folder}. Set RETRAIN=True to rerun the final-epoch validation protocol.")
            if reuse_completed and _complete(folder, dataset, graph):
                print(f"Reusing completed {len(SEEDS)}-run results: {LABELS[graph]}", flush=True)
                continue
        elif folder.is_dir() and any(folder.glob(f"{dataset}_{EXPERIMENT_TAG}_{graph}_seed_*.csv")):
            raise ValueError(f"Results have no graph provenance: {folder}. Choose a new output directory.")
        _write_json(manifest_path, manifest)
        print(f"Training {dataset} / {LABELS[graph]}: seeds {SEEDS}", flush=True)
        _stream_training_process(
            [sys.executable, "-u", "-m", "clinicalKG.run.common.classification_worker", dataset, graph], env)
        if not _complete(folder, dataset, graph):
            raise RuntimeError(f"Training did not produce complete predictions, histories and models: {folder}")


def _validate_saved_definitions(output_root, dataset, graphs):
    for graph in graphs:
        path = result_path(Path(output_root), dataset, graph, "run_manifest.json")
        if not path.is_file():
            raise FileNotFoundError(f"No provenance for this result: {path}. Use results from this notebook's training stage.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != dataset or manifest.get("graph") != graph:
            raise ValueError(f"Dataset/graph mismatch: {path}")
        if manifest.get("training_seeds") != list(SEEDS):
            raise ValueError(f"Seed configuration mismatch: {path}")
        if any(manifest.get(key) != value for key, value in EVALUATION_PROTOCOL.items()):
            raise ValueError(f"Old evaluation protocol: {path}. Rerun training with RETRAIN=True for final-epoch validation metrics.")
        if graph == "PrimeKG" and (manifest.get("primekg") or {}).get("scope") != "full_original_graph":
            raise ValueError(f"PrimeKG result is not marked as the original full graph: {path}")
        yield manifest


def saved_result_status(output_root, dataset, graphs=GRAPHS):
    """Inspect independently saved training files without running a model."""
    rows = []
    for graph in graphs:
        folder = Path(output_root)
        predictions = sum((result_path(folder, dataset, graph, f"seed_{seed}.csv")).is_file() for seed in SEEDS)
        histories = sum((result_path(folder, dataset, graph, f"history_seed_{seed}.csv")).is_file() for seed in SEEDS)
        checkpoints = sum((result_path(folder, dataset, graph, f"last_seed_{seed}.pt")).is_file() for seed in SEEDS)
        summary = (result_path(folder, dataset, graph, "summary.csv")).is_file()
        path = result_path(folder, dataset, graph, "run_manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        current = (manifest.get("training_seeds") == list(SEEDS)
                   and all(manifest.get(key) == value for key, value in EVALUATION_PROTOCOL.items()))
        rows.append({"graph": graph, "method": LABELS[graph], "predictions": f"{predictions}/{len(SEEDS)}",
                     "histories": f"{histories}/{len(SEEDS)}", "checkpoints": f"{checkpoints}/{len(SEEDS)}",
                     "summary": summary, "current_protocol": current,
                     "ready_for_statistics": current and summary and predictions == len(SEEDS),
                     "ready_for_curves": current and histories == len(SEEDS)})
    return rows


def metric_table(output_root, dataset, graphs=GRAPHS, write=True):
    import numpy as np
    import pandas as pd
    list(_validate_saved_definitions(output_root, dataset, graphs))
    rows = []
    for graph in graphs:
        path = result_path(Path(output_root), dataset, graph, "summary.csv")
        runs = pd.read_csv(path)
        if len(runs) != len(SEEDS) or set(runs["seed"]) != set(SEEDS):
            raise ValueError(f"Incomplete {len(SEEDS)}-run summary: {path}")
        row = {"dataset": dataset, "graph": graph, "method": LABELS[graph], "runs": len(runs)}
        for name in ("accuracy", "precision", "recall", "f1", "auc"):
            row[name + "_mean"] = float(np.mean(runs[name]))
            row[name + "_sd"] = float(np.std(runs[name], ddof=1))
        rows.append(row)
    table = pd.DataFrame(rows)
    if write:
        table.to_csv(result_path(Path(output_root), dataset, None, "metrics_mean_sd.csv"), index=False)
    return table


def run_statistics(output_root, dataset, metrics=("f1", "accuracy", "auc"),
                   resamples=10000, random_seed=2026, graphs=GRAPHS, reuse_saved=True):
    import numpy as np
    import pandas as pd
    from .kg_ablation_stats import read_predictions, aligned_arrays, paired_comparison
    if resamples < 100:
        raise ValueError("Use at least 100 resamples")
    if "DementiaHKG" not in graphs or "PrimeKG" not in graphs:
        raise ValueError("Include both DemRKG and Full PrimeKG")
    missing = []
    for graph in graphs:
        folder = Path(output_root)
        for filename in ("summary.csv", *(f"seed_{seed}.csv" for seed in SEEDS)):
            if not (result_path(folder, dataset, graph, filename)).is_file():
                missing.append(f"{LABELS[graph]}: {filename}")
    if missing:
        raise FileNotFoundError(
            f"{dataset} 的 {len(SEEDS)} 次训练结果尚未齐全。请等待对应 KG 的训练 cell 完成，"
            "再运行统计 cell；已完成的其他 KG 无需重跑。缺少：\n" + "\n".join(missing))
    manifests = list(_validate_saved_definitions(output_root, dataset, graphs))
    if len({entry["split_sha256"] for entry in manifests}) != 1:
        raise ValueError("Graph variants were trained using different patient partitions")
    fingerprints = {
        f"{graph}/{filename}": _hash(result_path(Path(output_root), dataset, graph, filename))
        for graph in graphs
        for filename in ("summary.csv", "run_manifest.json", *(f"seed_{seed}.csv" for seed in SEEDS))
    }
    cache_key = {"dataset": dataset, "metrics": list(metrics), "seeds": list(SEEDS),
                 "graphs": list(graphs), "resamples": resamples, "random_seed": random_seed,
                 "source_sha256": fingerprints}
    definition_path = result_path(Path(output_root), dataset, None, "analysis_definition.json")
    statistics_path = result_path(Path(output_root), dataset, None, "paired_statistics.csv")
    if reuse_saved and definition_path.is_file() and statistics_path.is_file():
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        if all(definition.get(key) == value for key, value in cache_key.items()):
            cached = pd.read_csv(statistics_path)
            expected = {(metric, LABELS[graph]) for metric in metrics for graph in graphs if graph != "DementiaHKG"}
            if (len(cached) == len(expected) and {tuple(row) for row in cached[["metric", "baseline"]].to_numpy()} == expected):
                print(f"{dataset}: 读取已保存的五指标统计结果，预测文件与统计参数一致。", flush=True)
                return metric_table(output_root, dataset, graphs), cached
    # Read all configured-seed predictions once, so each metric uses the same snapshot.
    predictions = {graph: read_predictions(Path(output_root), dataset, graph) for graph in graphs}
    reference = predictions["DementiaHKG"]
    comparisons = {graph: aligned_arrays(reference, predictions[graph], "patient")
                   for graph in graphs if graph != "DementiaHKG"}
    auc_mode = dataset_auc_mode(dataset)
    all_rows = []
    for metric in metrics:
        if metric not in ("f1", "accuracy", "auc", "precision", "recall"):
            raise ValueError(metric)
        rows = []
        for graph in graphs:
            if graph == "DementiaHKG":
                continue
            print(f"{dataset}: paired patient-level {metric}, DemRKG vs {LABELS[graph]} "
                  f"({resamples} bootstrap + {resamples} permutations)", flush=True)
            arrays = comparisons[graph]
            result = paired_comparison(*arrays, metric, auc_mode, resamples,
                                       np.random.default_rng(random_seed))
            rows.append({"dataset": dataset, "proposed": "DemRKG", "baseline": LABELS[graph],
                         "metric": metric, "unit": "patient", "auc_mode": auc_mode,
                         "resamples": resamples, **result})
        ordered = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (len(rows) - rank) * rows[index]["p_value"]))
            rows[index]["holm_p"] = previous
        pd.DataFrame(rows).to_csv(result_path(Path(output_root), dataset, None, f"paired_{metric}_patient.csv"), index=False)
        all_rows.extend(rows)
    table = pd.DataFrame(all_rows)
    table.to_csv(result_path(Path(output_root), dataset, None, "paired_statistics.csv"), index=False)
    _write_json(result_path(Path(output_root), dataset, None, "analysis_definition.json"),
                {"primekg_definition": PRIMEKG_DEFINITION, "dataset": dataset,
                 **cache_key, "unit": "patient",
                 "resamples": resamples, "auc_mode": auc_mode,
                 "difference": "mean across seeds of metric(DemRKG) - metric(baseline)",
                 "ci": "paired patient-cluster bootstrap, percentile 95% CI",
                 "p_value": "two-sided paired patient-cluster prediction swap permutation test",
                 "holm_family": "baseline comparisons within each dataset and metric"})
    return metric_table(output_root, dataset, graphs), table


def run_pipeline(dataset_choice=1, mode="all", output_root=None,
                 metrics=("f1", "accuracy", "auc"), resamples=10000):
    if dataset_choice not in DATASET_CHOICES:
        raise ValueError("DATASET_CHOICE must be 1=ADNI, 2=NACC, 3=AIBL, or 4=PPMI")
    if mode not in ("check", "train", "stats", "all"):
        raise ValueError("RUN_MODE must be check, train, stats, or all")
    dataset = DATASET_CHOICES[dataset_choice]
    output_root = default_output_root(dataset) if output_root is None else output_root
    if mode == "check":
        print(inspect_full_primekg())
        return check_inputs(dataset)
    if mode in ("train", "all"):
        train_classifiers(dataset, output_root)
    if mode in ("stats", "all"):
        return run_statistics(output_root, dataset, metrics, resamples)
    return metric_table(output_root, dataset), None


def plot_training_curves(output_root, dataset, graphs=GRAPHS):
    """Configured-seed mean ± sample SD for training and validation curves."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    list(_validate_saved_definitions(output_root, dataset, graphs))
    fig, axes = plt.subplots(len(graphs), 3, figsize=(16, 4 * len(graphs)), squeeze=False)
    for row, graph in enumerate(graphs):
        histories = []
        for seed in SEEDS:
            path = result_path(Path(output_root), dataset, graph, f"history_seed_{seed}.csv")
            if not path.is_file():
                raise FileNotFoundError(f"Training history not found: {path}. Use the updated training stage to save curves.")
            history = pd.read_csv(path).set_index("epoch")
            histories.append(history)
        for col, (metric, label) in enumerate((("loss", "Loss"), ("auc", "Macro AUC"), ("f1", "Macro F1"))):
            ax = axes[row, col]
            for prefix, color, legend in (("train", "tab:blue", "Train"), ("val", "tab:orange", "Validation")):
                values = pd.concat([history[prefix + "_" + metric] for history in histories], axis=1)
                mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1).fillna(0)
                ax.plot(values.index, mean, color=color, label=legend)
                ax.fill_between(values.index, mean - sd, mean + sd, color=color, alpha=0.15)
            ax.set(title=f"{LABELS[graph]} — {label}", xlabel="Epoch", ylabel=label)
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle(f"{dataset}: {len(SEEDS)} independent training runs (mean ± SD)", fontsize=14)
    fig.tight_layout()
    destination = result_path(Path(output_root), dataset, None, "training_curves.png")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved: {destination}")
    return fig
