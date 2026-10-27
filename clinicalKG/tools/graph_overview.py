"""Audit complete graph CSVs and draw type-balanced connected views.

Run from the repository root: python clinicalKG/tools/graph_overview.py
Statistics use every CSV row; layout uses an explicitly reported connected view.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clinicalKG.paths import CLINICALKG_ROOT, PROJECT_ROOT, raw_primekg_path, work_dir

COLORS = {
    "disease": "#E6194B", "gene/protein": "#3CB44B",
    "effect/phenotype": "#E8C819", "anatomy": "#4363D8",
    "biological_process": "#911EB4", "pathway": "#F58231",
    "cellular_component": "#42BFC4", "molecular_function": "#F032E6",
    "drug": "#A46A35", "exposure": "#7A8895",
}


def load_graph(path):
    names, relations, x_types, y_types = {}, {}, {}, {}
    source_nodes = set()
    heads, tails, rels = [], [], []
    columns = ["x_name", "y_name", "relation", "x_type", "y_type", "x_index", "y_index"]
    for frame in pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False,
                             chunksize=250_000):
        if (frame[["x_name", "y_name", "relation"]] == "").any().any():
            raise ValueError(f"Missing entity/relation name in {path}")
        for side, types in (("x", x_types), ("y", y_types)):
            for name, kind in frame[[side + "_name", side + "_type"]].drop_duplicates().itertuples(index=False, name=None):
                names.setdefault(name, len(names))
                types.setdefault(name, kind)
            source_nodes.update(frame[side + "_index"].unique())
        for relation in frame.relation.unique():
            relations.setdefault(relation, len(relations))
        heads.append(frame.x_name.map(names).to_numpy(dtype=np.int32))
        tails.append(frame.y_name.map(names).to_numpy(dtype=np.int32))
        rels.append(frame.relation.map(relations).to_numpy(dtype=np.int16))
    if not heads:
        raise ValueError(f"Empty graph: {path}")
    # Match the original Graph.ipynb: x-side types take precedence over y-side.
    node_types = {**y_types, **x_types}
    kinds = [node_types[name] for name in names]
    h, t, r = np.concatenate(heads), np.concatenate(tails), np.concatenate(rels)
    n = len(names)
    directed = h.astype(np.int64) * n + t
    unique_pairs = len(np.unique(directed))
    unique_triples = len(np.unique(directed * len(relations) + r))
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stats = {
        "source_csv": Path(path).relative_to(PROJECT_ROOT).as_posix()
            if Path(path).is_relative_to(PROJECT_ROOT) else str(Path(path)),
        "source_sha256": digest.hexdigest(),
        "entity_identity": "entity name, matching current DistMult indexing",
        "entities": n, "source_index_entities": len(source_nodes),
        "edge_records": len(h), "unique_directed_pairs": unique_pairs,
        "unique_directed_triples": unique_triples,
        "entity_types": len(set(kinds)), "relation_types": len(relations),
        "node_type_counts": dict(sorted(Counter(kinds).items())),
        "relation_record_counts": {name: int(value) for name, value in
            zip(relations, np.bincount(r, minlength=len(relations)))},
        "type_assignment": "first x-side type, falling back to first y-side type",
    }
    return list(names), kinds, h, t, r, list(relations), stats


def connected_view(names, kinds, h, t, r, relation_names, per_type, seed):
    n = len(names)
    adjacency = coo_matrix((np.ones(len(h), dtype=bool), (h, t)), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    count, labels = connected_components(adjacency, directed=False)
    largest = int(np.argmax(np.bincount(labels)))
    members = np.flatnonzero(labels == largest)
    degree = np.diff(adjacency.indptr)
    root = int(members[np.argmax(degree[members])])
    _, predecessor = breadth_first_order(adjacency, root, directed=False)
    groups = {}
    for index in members:
        groups.setdefault(kinds[index], []).append(int(index))
    rng, selected = random.Random(seed), {root}
    for kind in sorted(groups):
        candidates = sorted(groups[kind], key=lambda i: names[i])
        for target in rng.sample(candidates, min(per_type, len(candidates))):
            while target not in selected and target >= 0:
                selected.add(target)
                target = int(predecessor[target])
    mask = np.zeros(n, dtype=bool)
    mask[list(selected)] = True
    rows = np.flatnonzero(mask[h] & mask[t])
    graph = nx.Graph()
    graph.add_nodes_from(sorted(selected))
    graph.add_edges_from(zip(h[rows].tolist(), t[rows].tolist()))
    # Retain every relation type in the exported view, including mirror records.
    unique = np.unique((h[rows].astype(np.int64) * n + t[rows]) * len(relation_names) + r[rows])
    triples = [[int(value // len(relation_names) // n),
                int(value % len(relation_names)),
                int(value // len(relation_names) % n)] for value in unique]
    info = {
        "method": "largest connected component; highest-degree root; up to N targets per type; BFS shortest paths; induced view",
        "seed": seed, "targets_per_type": per_type,
        "full_connected_components": int(count), "largest_component_entities": len(members),
        "root": names[root], "entities": graph.number_of_nodes(),
        "undirected_pairs": graph.number_of_edges(), "edge_records": len(rows),
        "unique_directed_triples": len(unique),
        "node_type_counts": dict(sorted(Counter(kinds[i] for i in selected).items())),
        "relation_types": len(set(int(r[i]) for i in rows)),
    }
    export = {"nodes": [{"id": i, "name": names[i], "type": kinds[i]} for i in sorted(selected)],
              "relations": dict(enumerate(relation_names)), "triples": triples,
              "triple_columns": ["head_node_id", "relation_id", "tail_node_id"]}
    return graph, info, export


def draw_graph(graph, names, kinds, stats, info, output, title):
    positions = nx.spring_layout(graph, seed=2026, iterations=100)
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.73, top=0.89, bottom=0.09)
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#AAB6C0", alpha=0.24, width=0.45)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=20,
                           node_color=[COLORS.get(kinds[i], "#999999") for i in graph],
                           edgecolors="white", linewidths=0.15)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=21, fontweight="bold", x=0.04, ha="left", color="#223B55")
    fig.text(0.04, 0.915, "Type-balanced connected view · shortest-path routing", fontsize=12, color="#536779")
    fig.text(0.75, 0.85,
             f"COMPLETE INPUT GRAPH\n\n{stats['entities']:,} named entities\n"
             f"{stats['edge_records']:,} edge records\n{stats['relation_types']} relation types\n\n"
             f"DISPLAYED VIEW\n\n{info['entities']:,} entities\n{info['undirected_pairs']:,} undirected pairs\n"
             f"{info['unique_directed_triples']:,} directed triples", fontsize=11, va="top", linespacing=1.65,
             color="#223B55")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=COLORS.get(kind, "#999999"),
                      label=f"{kind} ({number:,})", markersize=7)
               for kind, number in stats["node_type_counts"].items()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.74, 0.10),
               frameon=False, title="Entity types · complete graph", fontsize=9, title_fontsize=10)
    fig.text(0.04, 0.035, "Statistics use every input row. Layout shows the connected view; node colours indicate entity types.",
             fontsize=10, color="#536779")
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    svg = output.with_suffix(".svg")
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=("all", "PrimeKGL", "DemRKL"), default="all")
    parser.add_argument("--targets-per-type", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.targets_per_type < 1:
        parser.error("--targets-per-type must be positive")
    sources = []
    if args.graph in ("all", "PrimeKGL"):
        sources.append(("PrimeKGL", "FullPrimeKG", raw_primekg_path(), "Full PrimeKG · shared across four cohorts"))
    if args.graph in ("all", "DemRKL"):
        for dataset in ("ADNI", "AIBL", "NACC", "PPMI"):
            filename = ("" if dataset == "ADNI" else dataset + "_") + "DementiaHKG_PrimeKG_Sub.csv"
            candidates = [work_dir(dataset, "DemRKL") / filename,
                          PROJECT_ROOT / dataset / "DementiaHKG" / filename]
            path = next((p for p in candidates if p.is_file()), candidates[0])
            sources.append(("DemRKL", dataset, path, dataset + " · DemRKG"))
    for method, name, path, title in sources:
        print(f"Auditing {name}: {path}", flush=True)
        names, kinds, h, t, r, relation_names, stats = load_graph(path)
        graph, view, export = connected_view(names, kinds, h, t, r, relation_names,
                                             args.targets_per_type, args.seed)
        stats["visualization"] = view
        folder = CLINICALKG_ROOT / method / "figures"
        folder.mkdir(exist_ok=True)
        (folder / (name + "-statistics.json")).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (folder / (name + "-view.json")).write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        draw_graph(graph, names, kinds, stats, view, folder / (name + "-graph"), title)
        print(f"{name}: {stats['entities']:,} entities, {stats['edge_records']:,} records; view {view['entities']} entities", flush=True)


if __name__ == "__main__":
    main()
