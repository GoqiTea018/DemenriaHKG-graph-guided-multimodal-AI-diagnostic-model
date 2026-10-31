"""Synthetic evaluation checks: no real patient diagnosis training."""
import ast
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from clinicalKG.run.common.evaluation import DATASET_CHOICES, GRAPHS, SEEDS, run_statistics, plot_training_curves, EVALUATION_PROTOCOL
from clinicalKG.run.common.classification_worker import training_cells, adapt_source
from clinicalKG.run.common.paths import result_path


class EvaluationWorkflow(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "tests" / ".runs"
        folder.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.output = Path(self.temp.name)
        self.assertTrue(self.output.resolve().is_relative_to(folder.resolve()))

    def tearDown(self):
        self.temp.cleanup()

    def fixtures(self):
        # All test files are synthetic; no values here are manuscript results.
        labels = [0, 1, 2] * 4
        for graph in GRAPHS:
            folder = self.output
            folder.mkdir(parents=True, exist_ok=True)
            manifest = {**EVALUATION_PROTOCOL, "dataset": "ADNI", "graph": graph, "split_sha256": "synthetic-common-split", "training_seeds": list(SEEDS),
                        "primekg": {"scope": "full_original_graph"} if graph == "PrimeKG" else None}
            (result_path(folder, "ADNI", graph, "run_manifest.json")).write_text(json.dumps(manifest), encoding="utf-8")
            with (result_path(folder, "ADNI", graph, "summary.csv")).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("seed", "best_epoch", "accuracy", "precision", "recall", "f1", "auc"))
                writer.writeheader()
                for i, seed in enumerate(SEEDS):
                    writer.writerow({"seed": seed, "best_epoch": 2,
                                     **dict.fromkeys(("accuracy", "precision", "recall", "f1", "auc"), 0.7 + 0.01 * i)})
            for seed in SEEDS:
                with (result_path(folder, "ADNI", graph, f"seed_{seed}.csv")).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("dataset", "graph", "seed", "record_id", "patient_id", "y_true", "y_pred",
                                     "prob_0", "prob_1", "prob_2"))
                    for i, label in enumerate(labels):
                        predicted = label if graph == "DementiaHKG" else (label + i % 2) % 3
                        probabilities = [0.05] * 3
                        probabilities[predicted] = 0.9
                        writer.writerow(("ADNI", graph, seed, f"r{i}", f"patient_{i // 2}", label, predicted, *probabilities))
                with (result_path(folder, "ADNI", graph, f"history_seed_{seed}.csv")).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("epoch", "train_loss", "train_auc", "train_f1", "val_loss", "val_auc", "val_f1"))
                    writer.writerows(((1, 1.2, 0.6, 0.5, 1.3, 0.55, 0.45),
                                      (2, 0.8, 0.7, 0.6, 0.9, 0.65, 0.55)))

    def test_all_twelve_classifiers_use_current_helpers_and_compile(self):
        self.assertEqual(DATASET_CHOICES, {1: "ADNI", 2: "NACC", 3: "AIBL", 4: "PPMI"})
        for dataset in DATASET_CHOICES.values():
            for graph in GRAPHS:
                for index, source in training_cells(dataset, graph):
                    tree = adapt_source(source, dataset, graph)
                    compile(tree, f"{dataset}/{graph}/{index}", "exec")
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            self.assertFalse(node.module.startswith("experiments."))

    def test_three_training_cells_dispatch_only_their_own_graph(self):
        notebook = json.loads((ROOT / "run" / "Run_Experiments.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"]
                   if cell["cell_type"] == "code" and "train_classifiers(DATASET" in "".join(cell["source"])]
        self.assertEqual(len(sources), 3)
        for graph, source in zip(GRAPHS, sources):
            trainer, checker = Mock(), Mock()
            namespace = {"DATASET": "ADNI", "OUTPUT_ROOT": self.output, "RUN_MODE": "train", "RETRAIN": False,
                         "train_classifiers": trainer, "check_inputs": checker}
            exec(source, namespace)
            trainer.assert_called_once_with("ADNI", self.output, graphs=(graph,), reuse_completed=True)
            checker.assert_not_called()
            trainer.reset_mock()
            namespace["RUN_MODE"] = "stats"
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            trainer.assert_not_called()

    def test_worker_logs_are_forwarded_to_notebook_stdout(self):
        from clinicalKG.run.common.evaluation import _stream_training_process
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            _stream_training_process([sys.executable, "-u", "-c",
                                      "import sys; print('epoch=1 train_loss=1.0',flush=True); "
                                      "print('epoch=2 val_loss=0.8',file=sys.stderr,flush=True)"], os.environ.copy())
        self.assertIn("epoch=1 train_loss=1.0", captured.getvalue())
        self.assertIn("epoch=2 val_loss=0.8", captured.getvalue())

    def test_customkg_cell_does_not_require_full_primekg_assets(self):
        from clinicalKG.run.common import evaluation
        with patch.object(evaluation, "inspect_full_primekg", side_effect=AssertionError("must not load Full PrimeKG")), \
                patch.object(evaluation, "check_inputs", return_value=[]), \
                patch.object(evaluation, "_graph_manifest", return_value={"graph": "CustomKG"}), \
                patch.object(evaluation, "_stream_training_process") as worker, \
                patch.object(evaluation, "_complete", return_value=True):
            evaluation.train_classifiers("ADNI", self.output, graphs=("CustomKG",))
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[1]["KG_ABLATION_SEEDS"], ",".join(map(str, SEEDS)))

    def test_statistics_provenance_and_curves(self):
        self.fixtures()
        performance, statistics = run_statistics(self.output, "ADNI", metrics=("f1",), resamples=100)
        self.assertEqual(len(performance), 3)
        self.assertEqual(len(statistics), 2)
        self.assertTrue((statistics["difference"] > 0).all())
        self.assertTrue((statistics["holm_p"] >= statistics["p_value"]).all())
        self.assertTrue((statistics["units"] == 6).all())
        self.assertIn("PrimeKG (Full original graph)", statistics["baseline"].values)
        from clinicalKG.run.common import kg_ablation_stats
        with patch.object(kg_ablation_stats, "paired_comparison", side_effect=AssertionError("cached results must not resample")):
            cached_performance, cached_statistics = run_statistics(self.output, "ADNI", metrics=("f1",), resamples=100)
        self.assertEqual(cached_statistics["holm_p"].tolist(), statistics["holm_p"].tolist())
        # A changed prediction fingerprint requires recalculation, even if the CSV stays valid.
        changed = result_path(self.output, "ADNI", "CustomKG", "seed_11.csv")
        with changed.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with patch.object(kg_ablation_stats, "paired_comparison", wraps=kg_ablation_stats.paired_comparison) as recalculate:
            run_statistics(self.output, "ADNI", metrics=("f1",), resamples=100)
        self.assertEqual(recalculate.call_count, 2)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with patch.object(plt, "show"):
            figure = plot_training_curves(self.output, "ADNI")
        self.assertEqual(len(figure.axes), 9)
        plt.close(figure)
        self.assertTrue((result_path(self.output, "ADNI", None, "training_curves.png")).is_file())
        path = result_path(self.output, "ADNI", "PrimeKG", "run_manifest.json")
        bad = json.loads(path.read_text(encoding="utf-8"))
        bad["primekg"]["scope"] = "old_seeded_two_hop"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not marked"):
            run_statistics(self.output, "ADNI", metrics=("f1",), resamples=100)

    def test_trainer_keeps_final_epoch_even_when_validation_loss_is_worse(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from clinicalKG.run.common.kg_ablation import run_ablation
        features = torch.eye(3).repeat(2, 1)
        labels = torch.tensor([0, 1, 2] * 2)
        dataset = TensorDataset(features, torch.zeros(6, 1, 1), torch.zeros(6, 1, dtype=torch.bool), labels)
        loader = DataLoader(dataset, batch_size=3)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x, *_):
                return self.linear(x)

        validation = iter([3.0, 1.0, 2.0] * 2)
        calls = []

        def train_step(model, *_):
            # Distinct states at every epoch expose accidental best-state reloads.
            epoch = len(calls) % 3 + 1
            with torch.no_grad():
                model.linear.bias.fill_(float(epoch))
            calls.append(epoch)
            return (1.0, 0.6, 0.5, 0.7, 0.8, 0.9)

        test_loader = DataLoader(TensorDataset(
            features, torch.zeros(6, 1, 1), torch.zeros(6, 1, dtype=torch.bool),
            torch.tensor([2, 0, 1] * 2)), batch_size=3)
        captured = io.StringIO()
        with patch.dict(os.environ, {"KG_ABLATION_OUTPUT": str(self.output)}), contextlib.redirect_stdout(captured):
            results, _ = run_ablation(
                dataset="ADNI", graph="PrimeKG", make_model=Model,
                train_loader=loader, val_loader=loader, test_loader=test_loader,
                train_epoch=train_step,
                validate_epoch=lambda *_: (next(validation), 0.6, 0.5, 0.7, 0.8, 0.9),
                epochs=3, learning_rate=0.001, weight_decay=0, device="cpu",
                test_record_ids=[f"r{i}" for i in range(6)],
                test_patient_ids=[f"p{i}" for i in range(6)],
                val_record_ids=[f"val{i}" for i in range(6)],
                val_patient_ids=[f"vp{i}" for i in range(6)], seeds=(11, 23))
        self.assertEqual([run["epoch"] for run in results], [3, 3])
        epoch_logs = [line for line in captured.getvalue().splitlines() if " epoch=" in line]
        self.assertEqual(len(epoch_logs), 6)
        for line in epoch_logs:
            for name in ("train_loss=", "train_accuracy=", "train_precision=", "train_recall=", "train_auc=", "train_f1=",
                         "val_loss=", "val_accuracy=", "val_precision=", "val_recall=", "val_auc=", "val_f1="):
                self.assertIn(name, line)
        for seed in (11, 23):
            folder = self.output
            with (result_path(folder, "ADNI", "PrimeKG", f"history_seed_{seed}.csv")).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual([float(row["val_loss"]) for row in rows], [3.0, 1.0, 2.0])
            self.assertEqual([float(row["val_accuracy"]) for row in rows], [0.9, 0.9, 0.9])
            self.assertEqual([float(row["train_precision"]) for row in rows], [0.8, 0.8, 0.8])
            state = torch.load(result_path(folder, "ADNI", "PrimeKG", f"last_seed_{seed}.pt"), map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(state["model_state_dict"]["linear.bias"], torch.full((3,), 3.0)))
            self.assertEqual(state["epoch"], 3)
            self.assertEqual(state["model_selection"], "final_epoch")
            self.assertEqual(state["evaluation_split"], "val")
            self.assertFalse((result_path(folder, "ADNI", "PrimeKG", f"best_seed_{seed}.pt")).exists())
            with (result_path(folder, "ADNI", "PrimeKG", f"seed_{seed}.csv")).open() as handle:
                predictions = list(csv.DictReader(handle))
            self.assertTrue(all(row["record_id"].startswith("val") for row in predictions))
            self.assertTrue(all(row["evaluation_split"] == "val" for row in predictions))
            self.assertEqual([int(row["y_true"]) for row in predictions], labels.tolist())

    def test_master_notebook_cells_execute_on_synthetic_saved_predictions(self):
        self.fixtures()
        notebook = json.loads((ROOT / "run" / "Run_Experiments.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["cells"][-1]["cell_type"], "code")
        self.assertIn("plot_training_curves", "".join(notebook["cells"][-1]["source"]))
        namespace = {"__name__": "__main__"}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with patch.object(plt, "show"), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                is_config = any(isinstance(node, ast.Assign) and
                                any(isinstance(target, ast.Name) and target.id == "DATASET_CHOICE"
                                    for target in node.targets)
                                for node in ast.parse(source).body)
                exec(compile(source, "Run_Experiments.ipynb", "exec"), namespace)
                if is_config:
                    namespace.update(DATASET_CHOICE=1, DATASET="ADNI", OUTPUT_ROOT=self.output,
                                     RUN_MODE="stats", RESAMPLES=100, METRICS=("f1",))
        self.assertEqual(namespace["DATASET"], "ADNI")
        self.assertEqual(len(namespace["statistics"]), 2)
        self.assertTrue((result_path(self.output, "ADNI", None, "training_curves.png")).is_file())
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
