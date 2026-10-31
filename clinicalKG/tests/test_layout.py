"""Check notebook entry points and the relocated data/output contract."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clinicalKG.run.common import paths
from clinicalKG.run.common.config import DATASETS, GRAPHS, SEEDS, EXPERIMENT_TAG, notebook_path, model_path, auc_mode
from clinicalKG.run.common.classification_worker import training_cells


class NotebookLayout(unittest.TestCase):
    def test_all_notebooks_compile_and_training_imports_resolve(self):
        for dataset in DATASETS:
            for graph in GRAPHS:
                entry = notebook_path(dataset, graph)
                expected = "DemRKG" if graph == "DementiaHKG" else graph
                self.assertEqual(entry.name, f"{dataset}三模态+{expected}.ipynb")
                self.assertTrue(model_path(dataset, graph).is_file())
                self.assertGreater(len(training_cells(dataset, graph)), 8)
        for entry in paths.CLINICALKG_ROOT.rglob("*.ipynb"):
            if any(part in ("analysis", "record", ".runs") for part in entry.parts):
                continue
            notebook = json.loads(entry.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                compile(source, f"{entry.name}:cell{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertNotEqual(node.module, "model1")
                        self.assertFalse(node.module.startswith("experiments."))
                        if node.module.startswith("clinicalKG."):
                            module = paths.CLINICALKG_ROOT.parent.joinpath(*node.module.split("."))
                            self.assertTrue(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file(), node.module)

    def test_run_dataset_precedes_absolute_fallback_including_nacc(self):
        test_root = paths.CLINICALKG_ROOT / "tests" / ".runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            root = Path(temporary)
            local, external = root / "run", root / "external"
            with patch.object(paths, "RUN_ROOT", local), patch.object(paths, "PROJECT_ROOT", external):
                for dataset in DATASETS:
                    self.assertEqual(paths.dataset_dir(dataset), external / dataset)
                    (local / dataset).mkdir(parents=True)
                    self.assertEqual(paths.dataset_dir(dataset), local / dataset)

    def test_each_bootstrap_runs_from_its_method_directory(self):
        previous = Path.cwd()
        try:
            for dataset in DATASETS:
                for graph in GRAPHS:
                    entry = notebook_path(dataset, graph)
                    os.chdir(entry.parent)
                    notebook = json.loads(entry.read_text(encoding="utf-8"))
                    bootstrap = next(cell for cell in notebook["cells"]
                                     if cell["cell_type"] == "code" and cell.get("metadata", {}).get("clinicalkg_stage") == "bootstrap")
                    namespace = {"__file__": str(entry)}
                    with patch.dict(os.environ, {}, clear=False):
                        exec("".join(bootstrap["source"]), namespace)
                    self.assertEqual(namespace["DATASET_DIR"], paths.dataset_dir(dataset))
                    self.assertEqual(namespace["CLINICALKG_ROOT"], paths.CLINICALKG_ROOT)
        finally:
            os.chdir(previous)

    def test_flat_filenames_keep_datasets_and_graphs_separate(self):
        root = Path("analysis")
        outputs = {paths.result_path(root, dataset, graph, "seed_11.csv")
                   for dataset in DATASETS for graph in GRAPHS}
        self.assertEqual(len(outputs), 12)
        self.assertTrue(all(path.parent == root for path in outputs))
        self.assertEqual(paths.result_path(root, "ADNI", None, "paired_statistics.csv"),
                         root / f"ADNI_{EXPERIMENT_TAG}_paired_statistics.csv")

    def test_requested_seeds_and_auc_are_shared_by_logs_and_final_scores(self):
        self.assertEqual(SEEDS, (51, 37, 67, 71, 109, 27, 11, 23, 97, 83))
        for dataset in DATASETS:
            self.assertEqual(auc_mode(dataset), "ovo" if dataset == "AIBL" else "ovr")
            for graph in GRAPHS:
                notebook = json.loads(notebook_path(dataset, graph).read_text(encoding="utf-8"))
                counts = {"roc_auc_score": 0, "run_ablation": 0}
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    for call in ast.walk(ast.parse("".join(cell["source"]))):
                        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                            continue
                        name = call.func.id
                        if name not in counts:
                            continue
                        keyword = "multi_class" if name == "roc_auc_score" else "auc_mode"
                        value = next(item.value for item in call.keywords if item.arg == keyword)
                        self.assertEqual(ast.unparse(value), "dataset_auc_mode(DATASET)")
                        counts[name] += 1
                self.assertGreaterEqual(counts["roc_auc_score"], 2)
                self.assertEqual(counts["run_ablation"], 1)

    def test_layout_alias_preserves_asset_and_protocol_checks(self):
        from clinicalKG.run.common import evaluation
        test_root = paths.CLINICALKG_ROOT / "tests" / ".runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            root = Path(temporary)
            record = root / "run" / "record"
            record.mkdir(parents=True)
            entry = {"dataset": "AIBL", "graph": "PrimeKG", "before_sha256": "old", "after_sha256": "new"}
            (record / "ALL_notebook_layout_migration_2026-10-31.json").write_text(json.dumps([entry]), encoding="utf-8")
            saved = {"dataset": "AIBL", "graph": "PrimeKG", "notebook_sha256": "old", "split_sha256": "split"}
            current = dict(saved, notebook_sha256="new")
            with patch.object(evaluation, "CLINICALKG_ROOT", root):
                self.assertTrue(evaluation._matches_saved_manifest(saved, current))
                self.assertFalse(evaluation._matches_saved_manifest(saved, dict(current, split_sha256="different")))
                self.assertFalse(evaluation._matches_saved_manifest(saved, dict(current, notebook_sha256="edited")))


if __name__ == "__main__":
    unittest.main()
