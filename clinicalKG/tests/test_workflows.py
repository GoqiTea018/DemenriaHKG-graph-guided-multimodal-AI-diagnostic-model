"""Small synthetic checks; never train the real PrimeKG or load patient MRI."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import contextlib
import io
import os
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

CLINICAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLINICAL.parent))
from clinicalKG.run.common.paths import asset_path, work_dir, PROJECT_ROOT, full_primekg_dir


class FullGraphCheckpoint(unittest.TestCase):
    def setUp(self):
        import torch
        from clinicalKG.PrimeKGL.full_primekg import train_full_primekg, load_graph
        self.torch = torch
        self.train = train_full_primekg
        self.load_graph = load_graph
        test_root = CLINICAL / "tests" / ".runs"
        test_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary.name)
        assert self.root.resolve().is_relative_to(test_root.resolve())
        self.graph = self.root / "kg.csv"
        # Includes drug/exposure relations and a disconnected component: all must be retained.
        rows = ["x_name,relation,y_name,x_type,y_type"]
        for i in range(14):
            rows.append(f"e{i % 7},indication,e{(i+1) % 7},drug,exposure")
        rows.append("remote_a,off-label use,remote_b,drug,disease")
        self.graph.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.settings = dict(epochs=2, batch_size=2, embed_dim=4, lr=0.005,
                             device="cpu", seed=123, save_every_steps=2)

    def tearDown(self):
        self.temporary.cleanup()

    def state(self, folder):
        return self.torch.load(folder / "PrimeKG-DistMult-last.pt", weights_only=False)

    def test_full_graph_retains_all_records_and_relations(self):
        triplets, entities, relations = self.load_graph(self.graph)
        self.assertEqual(len(triplets), 15)
        self.assertIn("remote_a", entities)
        self.assertIn("off-label use", relations)

    def test_sigint_mid_epoch_and_resume_equal_uninterrupted(self):
        baseline, interrupted = self.root / "baseline", self.root / "interrupted"
        self.train(self.graph, baseline, **self.settings)
        original_step = self.torch.optim.Adam.step
        calls = 0

        def stop_after_third(optimizer, *args, **kwargs):
            nonlocal calls
            result = original_step(optimizer, *args, **kwargs)
            calls += 1
            if calls == 3:
                signal.raise_signal(signal.SIGINT)
            return result

        with patch.object(self.torch.optim.Adam, "step", stop_after_third):
            result = self.train(self.graph, interrupted, **self.settings)
        self.assertTrue(result["interrupted"])
        saved = self.state(interrupted)
        self.assertEqual(saved["next_row"], 6)
        self.assertEqual(saved["global_step"], 3)
        self.assertEqual(saved["epoch"], 0)
        import numpy as np
        np.testing.assert_array_equal(np.load(interrupted / "PrimeKG-Embeddings.npy"),
                                      saved["model_state_dict"]["ent_emb.weight"].numpy())
        self.train(self.graph, interrupted, resume=True, **self.settings)
        expected, actual = self.state(baseline), self.state(interrupted)
        self.assertEqual(expected["global_step"], actual["global_step"])
        self.assertEqual(expected["loss_history"], actual["loss_history"])
        for key, value in expected["model_state_dict"].items():
            self.assertTrue(self.torch.equal(value, actual["model_state_dict"][key]), key)
        for index, state in expected["optimizer_state_dict"]["state"].items():
            for key, value in state.items():
                other = actual["optimizer_state_dict"]["state"][index][key]
                self.assertTrue(self.torch.equal(value, other), key)
        before = (interrupted / "PrimeKG-DistMult-last.pt").read_bytes()
        with self.assertRaises(ValueError):
            self.train(self.graph, interrupted, resume=True, **{**self.settings, "batch_size": 3})
        self.assertEqual(before, (interrupted / "PrimeKG-DistMult-last.pt").read_bytes())
        with self.assertRaises(FileExistsError):
            self.train(self.graph, interrupted, **self.settings)

    def test_sigterm_saves_current_step(self):
        original_step = self.torch.optim.Adam.step

        def stop(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            signal.raise_signal(signal.SIGTERM)
            return result

        output = self.root / "terminated"
        with patch.object(self.torch.optim.Adam, "step", stop):
            result = self.train(self.graph, output, **self.settings)
        self.assertTrue(result["interrupted"])
        self.assertEqual(self.state(output)["global_step"], 1)
        self.assertEqual(json.loads((output / "PrimeKG-TrainingState.json").read_text())["reason"], "interrupted")


if __name__ == "__main__":
    unittest.main()
