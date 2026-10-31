"""Run one cohort / graph classifier in a fresh process using current KG assets."""
from __future__ import annotations
import argparse
import ast
import json
import os
import sys
from pathlib import Path

from .paths import CLINICALKG_ROOT, PROJECT_ROOT, asset_path, dataset_dir
from .config import asset_names, notebook_path, model_path, GRAPHS, DATASETS


def training_cells(dataset, graph):
    path = notebook_path(dataset, graph)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        cells.append((index, source))
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "run_ablation" for node in ast.walk(tree)):
            return cells
    raise ValueError(f"No training cell in {path}")


def adapt_source(source, dataset, graph):
    tree = ast.parse(source)
    # Reuse predictions already collected by each epoch function; no extra forward pass.
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in ("train_epoch", "validate_epoch"):
            for node in ast.walk(function):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 5:
                    node.value.elts.append(ast.Call(func=ast.Name(id="accuracy_score", ctx=ast.Load()),
                                                   args=[ast.Name(id="all_labels", ctx=ast.Load()),
                                                         ast.Name(id="all_preds", ctx=ast.Load())], keywords=[]))
    names = asset_names(dataset, graph)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("experiments."):
            node.module = "clinicalKG.run.common." + node.module.removeprefix("experiments.")
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                node.value = ast.Constant(asset_path(dataset, graph, names[name]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_ablation":
            for keyword in node.keywords:
                if keyword.arg == "graph":
                    keyword.value = ast.Constant(graph)
            existing = {keyword.arg for keyword in node.keywords}
            for name, values in (("val_record_ids", "record_ids"), ("val_patient_ids", "patient_ids")):
                if name not in existing:
                    node.keywords.append(ast.keyword(
                        arg=name, value=ast.parse(f"{values}[_clinical_val_indices]", mode="eval").body))
    return ast.fix_missing_locations(tree)


def execute(dataset, graph):
    from .experiment_split import cohort_frame, original_record_indices
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(dataset_dir(dataset))
    model_dir = model_path(dataset, graph).parent
    sys.path.insert(0, str(model_dir))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path(dataset, graph)),
                 "CLINICALKG_ROOT": CLINICALKG_ROOT,
                 "_clinical_val_indices": original_record_indices(cohort_frame(dataset)["_label"])[1]}
    for index, source in training_cells(dataset, graph):
        print(f"[{dataset}/{graph}] notebook cell {index}", flush=True)
        exec(compile(adapt_source(source, dataset, graph),
                     f"{dataset}/{graph}:cell{index}", "exec"), namespace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("graph", choices=GRAPHS)
    args = parser.parse_args()
    execute(args.dataset, args.graph)
