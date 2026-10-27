"""Command-line entry points for the copied, editable notebooks."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import os


def run_cells(notebook_path, stages=None, *, dry_run=False, stop_after_training=False):
    notebook_path = Path(notebook_path).resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    selected = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        stage = cell.get("metadata", {}).get("clinicalkg_stage")
        if stages is not None and stage not in {*stages, "bootstrap"}:
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source, filename=f"{notebook_path.name}:cell{index}")
        selected.append((index, stage))
        print(f"{notebook_path.name}: cell {index}, stage={stage}", flush=True)
        if not dry_run:
            exec(compile(tree, f"{notebook_path.name}:cell{index}", "exec"), namespace)
        if stop_after_training and any(isinstance(node, ast.Call)
                                      and isinstance(node.func, ast.Name)
                                      and node.func.id == "run_ablation" for node in ast.walk(tree)):
            break
    return selected


def build_main(notebook_path, method):
    parser = argparse.ArgumentParser(description="Run copied graph/mapping/KGE notebook stages")
    parser.add_argument("--stage", choices=("all", "graph", "map", "kge", "plot"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Compile and list cells; do not execute")
    if method == "PrimeKGL":
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--raw-kg", type=Path, default=None)
        parser.add_argument("--device", default=None)
        parser.add_argument("--save-every-steps", type=int, default=1000)
        parser.add_argument("--log-every-steps", type=int, default=100)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    if method == "PrimeKGL" and args.raw_kg:
        os.environ["CLINICALKG_RAW_KG"] = str(args.raw_kg.resolve())
    stages = {args.stage} if args.stage != "all" else {"graph", "map", "kge", "plot"}
    if method == "PrimeKGL" and not args.dry_run:
        # The shared trainer has no dependency on patient CSVs or MRI files.
        if stages & {"graph", "map"}:
            run_cells(notebook_path, stages & {"graph", "map"})
        if "kge" in stages:
            from full_primekg import train_full_primekg
            train_full_primekg(raw_kg=args.raw_kg, epochs=args.epochs,
                               batch_size=args.batch_size, resume=args.resume,
                               device=args.device, save_every_steps=args.save_every_steps,
                               log_every_steps=args.log_every_steps)
        if "plot" in stages or "kge" in stages:
            from full_primekg import plot_loss
            plot_loss()
        return
    run_cells(notebook_path, stages, dry_run=args.dry_run)


def train_main(notebook_path):
    parser = argparse.ArgumentParser(description="Run the original five-run multimodal classifier")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Same stopping point as the original experiments.worker; notebook retains plots/profiling cells.
    run_cells(notebook_path, dry_run=args.dry_run, stop_after_training=True)
