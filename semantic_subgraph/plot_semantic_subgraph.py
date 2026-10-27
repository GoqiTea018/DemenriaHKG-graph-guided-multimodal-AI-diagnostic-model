"""从已有 DemRKG CSV 绘制带实体名、类型和关系标签的语义子图。

在 semantic_subgraph 目录运行：
    python plot_semantic_subgraph.py
    python plot_semantic_subgraph.py --dataset ADNI
    python plot_semantic_subgraph.py --dataset NACC --output example.png

默认仅生成 NACC 示例 PNG。所有节点、关系均从所选数据集的 CSV 读取。
图中展示人工选取的局部连接；布局用于阅读实体和关系，不表示嵌入坐标。
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# 与现有白底交互概览的颜色一致，便于在论文中共用图例。
TYPES = {
    "disease": ("Disease", "#E6194B"),
    "gene/protein": ("Gene/protein", "#3CB44B"),
    "effect/phenotype": ("Phenotype", "#FFE119"),
    "anatomy": ("Anatomy", "#4363D8"),
    "biological_process": ("Biological process", "#911EB4"),
    "pathway": ("Pathway", "#F58231"),
    "cellular_component": ("Cellular component", "#42D4F4"),
    "molecular_function": ("Molecular function", "#F032E6"),
}


@dataclass(frozen=True)
class Entity:
    index: str
    name: str
    kind: str


@dataclass(frozen=True)
class Link:
    source: Entity
    target: Entity
    relation: str
    display: str


def read_graph(path: Path) -> list[Link]:
    """保留实体索引和每条原始关系，避免按名称建简单图时覆盖关系。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [
            Link(
                Entity(row["x_index"], row["x_name"], row["x_type"]),
                Entity(row["y_index"], row["y_name"], row["y_type"]),
                row["relation"],
                row["display_relation"],
            )
            for row in csv.DictReader(stream)
        ]


def select_example(links: list[Link]) -> tuple[list[Link], dict[Entity, tuple[float, float]]]:
    """选取可读的关系骨架；仅选已有边，并将反向重复记录合并显示。"""
    entities = {entity for link in links for entity in (link.source, link.target)}

    def named(name: str, kind: str) -> Entity:
        matches = [e for e in entities if e.name == name and e.kind == kind]
        if len(matches) != 1:
            raise ValueError(f"Expected one entity for {name!r} ({kind}), found {len(matches)}")
        return matches[0]

    def existing(a: Entity, b: Entity, relation: str) -> Link:
        matches = [
            link for link in links
            if link.relation == relation
            and ((link.source == a and link.target == b) or (link.source == b and link.target == a))
        ]
        if not matches:
            raise ValueError(f"Missing source relation: {a.name} / {relation} / {b.name}")
        # 对同一对实体的这一关系，仅展示一次，仍保留一条真实源记录。
        return min(matches, key=lambda link: (link.source.index, link.target.index))

    disease = named("Alzheimer disease", "disease")
    app = named("APP", "gene/protein")
    selected = [existing(disease, app, "disease_protein")]
    def connected_protein(preferred: tuple[str, ...], excluded: set[Entity]) -> tuple[Entity, Link]:
        candidates = sorted((e for e in entities if e.kind == "gene/protein" and e not in excluded), key=lambda e: (preferred.index(e.name) if e.name in preferred else len(preferred), e.name, e.index))
        for protein in candidates:
            for anchor, relation in ((app, "protein_protein"), (disease, "disease_protein")):
                try:
                    return protein, existing(anchor, protein, relation)
                except ValueError:
                    continue
        raise ValueError("No connected gene/protein available for the example")

    upper, upper_link = connected_protein(("APOE", "SNCA", "ACHE", "LRRK2"), {app})
    lower, lower_link = connected_protein(("MAPT", "LRRK2", "PSEN1"), {app, upper})
    selected.extend((upper_link, lower_link))
    positions = {disease: (-3.6, 0.0), app: (0.8, 0.0), upper: (0.8, 3.2), lower: (0.8, -3.2)}

    phenotype_links = {}
    for link in links:
        if link.relation != "disease_phenotype_positive":
            continue
        other = link.target if link.source == disease else link.source if link.target == disease else None
        if other is not None and other.kind == "effect/phenotype":
            phenotype_links[other] = existing(disease, other, link.relation)
    used_phenotypes = set()
    for name, y in (("Memory impairment", 3.2), ("Cognitive impairment", 0.0), ("Aphasia", -3.2)):
        candidates = [entity for entity in phenotype_links if entity not in used_phenotypes]
        if not candidates:
            break
        phenotype = min(candidates, key=lambda entity: (entity.name != name, len(entity.name), entity.name, entity.index))
        used_phenotypes.add(phenotype)
        selected.append(phenotype_links[phenotype])
        positions[phenotype] = (-9.4, y)

    # 首选与认知和神经退行性疾病相关且名称简洁的实体。
    # 若该数据集没有首选连接，就从同类型的真实邻居中确定性地选择一个。
    contexts = [
        (upper, "molecular_function", "molfunc_protein", ("amyloid-beta binding", "cholesterol binding"), 4.0),
        (app, "pathway", "pathway_protein", ("Amyloid fiber formation", "The NLRP3 inflammasome"), 2.0),
        (app, "anatomy", "anatomy_protein_present", ("cerebral cortex", "brain", "hippocampus"), 0.0),
        (app, "cellular_component", "cellcomp_protein", ("synapse", "axon", "extracellular space"), -2.0),
        (lower, "biological_process", "bioprocess_protein", ("memory", "axonal transport", "learning"), -4.0),
    ]
    for anchor, kind, relation, preferred, y in contexts:
        neighbors = set()
        for link in links:
            if link.relation != relation:
                continue
            if link.source == anchor and link.target.kind == kind:
                neighbors.add(link.target)
            elif link.target == anchor and link.source.kind == kind:
                neighbors.add(link.source)
        if not neighbors:
            print(f"Skipped {kind}: no {relation} neighbor of {anchor.name} in this dataset")
            continue
        def rank(entity: Entity) -> tuple:
            preference = preferred.index(entity.name) if entity.name in preferred else len(preferred)
            return preference, len(entity.name), entity.name, entity.index
        chosen = min(neighbors, key=rank)
        selected.append(existing(anchor, chosen, relation))
        positions[chosen] = (6.1, y)
    return selected, positions


def draw_schematic(dataset: str, selected: list[Link], positions: dict[Entity, tuple[float, float]], output: Path, dpi: int) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.83, bottom=0.19)
    ax.set(xlim=(-11.6, 8.6), ylim=(-5.2, 5.2))
    ax.axis("off")
    fig.text(0.5, 0.945, f"{dataset} DemRKG: an illustrative semantic subgraph", ha="center", fontsize=23, weight="bold", color="#172B4D")
    fig.text(0.5, 0.900, f"{len(positions)} selected entities  |  {len(selected)} selected relations", ha="center", fontsize=13, color="#536273")

    for x, heading in ((-9.4, "CLINICAL PHENOTYPES"), (-3.6, "DISEASE"), (0.8, "GENES / PROTEINS"), (6.1, "BIOLOGICAL CONTEXT")):
        ax.text(x, 5.08, heading, ha="center", va="bottom", color="#65758B", fontsize=11, weight="bold")

    # 节点以彩色边框和同色圆点表示类型，实体名称保持源 CSV 中的拼写。
    boxes = {}
    for entity, (x, y) in positions.items():
        title, color = TYPES[entity.kind]
        width = 3.35 if entity.kind not in ("gene/protein", "disease") else (2.9 if entity.kind == "disease" else 2.25)
        pale = tuple(0.09 * component + 0.91 for component in to_rgb(color))
        patch = FancyBboxPatch((x - width / 2, y - 0.50), width, 1.0, boxstyle="round,pad=0.025,rounding_size=0.13", facecolor=pale, edgecolor=color, linewidth=1.9, zorder=3)
        ax.add_patch(patch)
        boxes[entity] = patch
        ax.text(x, y + 0.13, textwrap.fill(entity.name, 24), ha="center", va="center", fontsize=13, weight="bold", color="#172B4D", zorder=4)
        ax.text(x, y - 0.28, title, ha="center", va="center", fontsize=10, color="#536273", zorder=4)

    # 使用无箭头连线展示关系，原始类型和显示名称同时标注。
    # CSV 含镜像记录，因此图上的位置和连线方向不承担因果含义。
    for link in selected:
        a, b = positions[link.source], positions[link.target]
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-", "color": "#8292A6", "lw": 1.5, "patchA": boxes[link.source], "patchB": boxes[link.target], "shrinkA": 3, "shrinkB": 3}, zorder=1)
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if a[0] == b[0]:
            x += 0.05
        label = "protein interaction" if link.display == "ppi" else link.display
        ax.text(x, y + 0.12, label, ha="center", va="bottom", fontsize=10.5, color="#354760", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=5)
        ax.text(x, y - 0.12, link.relation, ha="center", va="top", fontsize=8.3, color="#65758B", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3}, zorder=5)

    present = {entity.kind for entity in positions}
    legend = [Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor=color, markeredgecolor="#68778A", markeredgewidth=0.4, label=title) for kind, (title, color) in TYPES.items() if kind in present]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=4, frameon=False, fontsize=11.5, columnspacing=2.0, handletextpad=0.5)
    fig.text(0.5, 0.045, "Selected connections from the cohort-specific DemRKG; edge labels retain the source relation types.", ha="center", fontsize=10.5, color="#536273")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def induced_links(links: list[Link], entities: set[Entity]) -> list[Link]:
    """展示所选实体之间的全部关系，按实体对及关系类型合并镜像记录。"""
    unique = {}
    for link in links:
        if link.source in entities and link.target in entities:
            key = (tuple(sorted((link.source.index, link.target.index))), link.relation)
            unique.setdefault(key, link)
    return sorted(unique.values(), key=lambda link: (link.relation, link.source.index, link.target.index))


def draw_network(dataset: str, links: list[Link], entities: set[Entity], output: Path, dpi: int) -> None:
    """白底、彩色圆点、浅灰连线，与原有整体图保持一致。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    graph = nx.Graph()
    graph.add_nodes_from(sorted(entities, key=lambda entity: (entity.kind, entity.name, entity.index)))
    graph.add_edges_from((link.source, link.target) for link in links)
    # 降低高连接度核心之间的布局吸引力，给实体名和边编号留出空间。
    # 此权重只用于排版，数据中的关系及其数量保持原样。
    for a, b in graph.edges:
        graph[a][b]["layout_weight"] = 1 / (graph.degree[a] * graph.degree[b]) ** 0.5
    positions = nx.spring_layout(graph, seed=42, k=1.4, iterations=400, weight="layout_weight")
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    ax = fig.add_axes((0.025, 0.18, 0.70, 0.70))
    key_ax = fig.add_axes((0.75, 0.20, 0.24, 0.66))
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal")
    ax.axis("off")
    key_ax.axis("off")
    fig.text(0.5, 0.955, f"{dataset} DemRKG: a labeled local subgraph", ha="center", fontsize=22, weight="bold", color="#263444")
    fig.text(0.5, 0.915, f"{len(entities)} entities  |  {len(links)} relations among the selected entities", ha="center", fontsize=12, color="#647181")
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#bdc3c7", width=1.05, alpha=0.72)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=[TYPES[node.kind][1] for node in graph], node_size=230, linewidths=0.8, edgecolors="#718096")

    # 优先放置中心实体标签，尝试八个方位，减少文字与节点的重叠。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    from matplotlib.transforms import Bbox
    for entity in graph:
        px, py = ax.transData.transform(positions[entity])
        occupied.append(Bbox.from_bounds(px - 11, py - 11, 22, 22))
    for entity in sorted(graph, key=lambda node: (-graph.degree[node], node.name)):
        x, y = positions[entity]
        candidates = [(0, 17, "center", "bottom"), (0, -17, "center", "top"), (17, 0, "left", "center"), (-17, 0, "right", "center"), (14, 14, "left", "bottom"), (-14, 14, "right", "bottom"), (14, -14, "left", "top"), (-14, -14, "right", "top")]
        choices = []
        for order, (dx, dy, ha, va) in enumerate(candidates):
            annotation = ax.annotate(textwrap.fill(entity.name, 23), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=12, weight="semibold", color="#263444", bbox={"facecolor": "white", "alpha": 0.93, "edgecolor": "none", "pad": 1.8}, zorder=5)
            bbox = annotation.get_window_extent(renderer).expanded(1.04, 1.10)
            cost = sum(bbox.overlaps(other) for other in occupied)
            choices.append((cost, order, dx, dy, ha, va))
            annotation.remove()
        _, _, dx, dy, ha, va = min(choices)
        annotation = ax.annotate(textwrap.fill(entity.name, 23), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=12, weight="semibold", color="#263444", bbox={"facecolor": "white", "alpha": 0.93, "edgecolor": "none", "pad": 1.8}, zorder=5)
        occupied.append(annotation.get_window_extent(renderer).expanded(1.04, 1.10))

    relations = sorted({link.relation for link in links})
    codes = {relation: f"R{index + 1}" for index, relation in enumerate(relations)}
    # 每条边显示关系编号，右侧逐一解释；重复关系共享编号。
    # 与仅保留关系骨架相比，这里同时显示选中实体之间的所有原始关系类型。
    for link in links:
        a, b = positions[link.source], positions[link.target]
        choices = []
        for order, t in enumerate((0.5, 0.35, 0.65, 0.25, 0.75, 0.42, 0.58, 0.20, 0.80)):
            x, y = a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t
            label = ax.text(x, y, codes[link.relation], ha="center", va="center", fontsize=8.7, color="#647181", zorder=4)
            bbox = label.get_window_extent(renderer).expanded(1.4, 1.4)
            choices.append((sum(bbox.overlaps(other) for other in occupied), order, x, y))
            label.remove()
        _, _, x, y = min(choices)
        label = ax.text(x, y, codes[link.relation], ha="center", va="center", fontsize=8.7, color="#647181", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2}, zorder=4)
        occupied.append(label.get_window_extent(renderer).expanded(1.4, 1.4))

    key_ax.text(0, 1.0, "RELATION KEY", fontsize=12, weight="bold", color="#263444", va="top")
    for index, relation in enumerate(relations):
        sample = next(link for link in links if link.relation == relation)
        label = "protein interaction" if sample.display == "ppi" else sample.display
        y = 0.925 - index * 0.089
        key_ax.text(0, y, codes[relation], fontsize=10.5, weight="bold", color="#536273", va="top")
        key_ax.text(0.12, y, label, fontsize=11, color="#263444", va="top")
        key_ax.text(0.12, y - 0.029, relation, fontsize=8.4, color="#647181", va="top")
    present = {entity.kind for entity in entities}
    legend = [Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=color, markeredgecolor="#718096", markeredgewidth=0.5, label=title) for kind, (title, color) in TYPES.items() if kind in present]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False, fontsize=11.5, columnspacing=1.8)
    fig.text(0.5, 0.025, "Node colors follow the graph overview. Reverse duplicate records are displayed once per relation type.", ha="center", fontsize=9.5, color="#647181")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("NACC", "ADNI", "AIBL", "PPMI"), default="NACC")
    parser.add_argument("--output", type=Path, help="PNG 输出路径；默认保存到脚本旁的 output 目录")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--style", choices=("network", "schematic"), default="network", help="network 与原概览同风格；schematic 为分组关系骨架")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    folder = ROOT / args.dataset / "DementiaHKG"
    csv_name = "DementiaHKG_PrimeKG_Sub.csv" if args.dataset == "ADNI" else f"{args.dataset}_DementiaHKG_PrimeKG_Sub.csv"
    output = args.output or SCRIPT_DIR / "output" / f"{args.dataset}-DemRKG-semantic-subgraph.png"
    if output.suffix.lower() != ".png":
        parser.error("--output must end with .png")
    source_links = read_graph(folder / csv_name)
    selected, positions = select_example(source_links)
    if args.style == "network":
        selected = induced_links(source_links, set(positions))
        draw_network(args.dataset, selected, set(positions), output, args.dpi)
    else:
        draw_schematic(args.dataset, selected, positions, output, args.dpi)
    print(f"Saved: {output.resolve()}")
    print(f"Entities: {len(positions)}; displayed relations: {len(selected)}")
    for link in selected:
        print(f"  {link.source.name} --[{link.relation}]-- {link.target.name}")


if __name__ == "__main__":
    main()
