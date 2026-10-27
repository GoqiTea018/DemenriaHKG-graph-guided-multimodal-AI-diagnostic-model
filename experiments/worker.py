"""Execute one trusted project notebook through its five-run training cell."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

from .config import DATASETS, GRAPHS, ROOT, asset_names, notebook_path, resolve_graph_asset


def training_cells(dataset, graph):
    path = notebook_path(dataset, graph)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=f"{path.name}:cell{index}")
        cells.append((index, source))
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "run_ablation" for node in ast.walk(tree)):
            return cells
    raise ValueError(f"No five-run training cell in {path}")


def adapt_source(source, dataset, graph):
    """Set only KG filenames and the export graph name in a shared template."""
    tree = ast.parse(source)
    assets = asset_names(dataset, graph)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in assets:
                node.value = ast.Constant(resolve_graph_asset(dataset, graph, assets[name]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_ablation":
            for keyword in node.keywords:
                if keyword.arg == "graph":
                    keyword.value = ast.Constant(graph)
    return ast.fix_missing_locations(tree)


def execute(dataset, graph):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(ROOT / dataset)
    # model1 imports resolve to this cohort, with each worker in a fresh process.
    import sys
    sys.path.insert(0, str(ROOT / dataset))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path(dataset, graph))}
    for index, source in training_cells(dataset, graph):
        print(f"[{dataset}/{graph}] notebook cell {index}", flush=True)
        tree = adapt_source(source, dataset, graph)
        exec(compile(tree, f"{dataset}/{graph}:cell{index}", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("graph", choices=GRAPHS)
    args = parser.parse_args()
    execute(args.dataset, args.graph)


if __name__ == "__main__":
    main()
