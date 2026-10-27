"""Exact graph counts and connected views on a tiny typed CSV."""
import tempfile
from pathlib import Path
import unittest

import networkx as nx
import pandas as pd

from clinicalKG.tools.graph_overview import load_graph, connected_view


class GraphOverview(unittest.TestCase):
    def test_duplicate_records_parallel_relations_and_name_identity(self):
        rows = [
            ("A", "B", "r1", "disease", "gene/protein", "0", "1"),
            ("A", "B", "r1", "disease", "gene/protein", "0", "1"),
            ("A", "B", "r2", "disease", "gene/protein", "0", "1"),
            ("B", "A", "r1", "gene/protein", "disease", "1", "0"),
            ("B", "C", "r1", "gene/protein", "anatomy", "1", "2"),
            ("C", "A", "r1", "anatomy", "disease", "2", "3"),
            ("D", "D", "r1", "drug", "drug", "4", "4"),
        ]
        test_root = Path(__file__).resolve().parent / ".runs"
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            path = Path(folder) / "graph.csv"
            pd.DataFrame(rows, columns=["x_name", "y_name", "relation", "x_type", "y_type",
                                        "x_index", "y_index"]).to_csv(path, index=False)
            names, kinds, h, t, r, relations, stats = load_graph(path)
        self.assertEqual(stats["entities"], 4)
        self.assertEqual(stats["source_index_entities"], 5)
        self.assertEqual(stats["edge_records"], 7)
        self.assertEqual(stats["unique_directed_pairs"], 5)
        self.assertEqual(stats["unique_directed_triples"], 6)
        self.assertEqual(sum(stats["node_type_counts"].values()), 4)
        self.assertEqual(stats["relation_record_counts"], {"r1": 6, "r2": 1})
        graph, info, export = connected_view(names, kinds, h, t, r, relations, 60, 42)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(info["entities"], 3)
        self.assertEqual(info["edge_records"], 6)
        self.assertEqual(info["unique_directed_triples"], 5)
        self.assertEqual(len(export["triples"]), 5)
        _, info_again, export_again = connected_view(names, kinds, h, t, r, relations, 60, 42)
        self.assertEqual(info, info_again)
        self.assertEqual(export, export_again)


if __name__ == "__main__":
    unittest.main()
