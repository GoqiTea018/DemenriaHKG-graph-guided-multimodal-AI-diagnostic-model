"""Create copies from the original project, recording their source hashes.

Run only when deliberately regenerating the copies; never edits original files.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

CLINICAL = Path(__file__).resolve().parents[1]
ROOT = CLINICAL.parent
DATASETS = ("ADNI", "AIBL", "NACC", "PPMI")
manifest = []


def record(source, destination, note="copied"):
    manifest.append({"source": str(source.relative_to(ROOT)).replace("\\", "/"),
                     "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "destination": str(destination.relative_to(CLINICAL)).replace("\\", "/"),
                     "changes": note})


def bootstrap(dataset, method, training=False):
    return f'''# Paths relative to the main folder; portable between Windows and Linux.
from pathlib import Path
import os
import sys
_starts = [Path.cwd(), Path(globals().get("__file__", Path.cwd() / "notebook")).resolve().parent]
CLINICALKG_ROOT = next(p for start in _starts for p in (start, *start.parents)
                      if (p / "full_primekg.py").is_file() and (p / "paths.py").is_file())
for _path in (CLINICALKG_ROOT.parent, CLINICALKG_ROOT):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))
from clinicalKG.paths import PROJECT_ROOT, SHARED_PRIMEKG, raw_primekg_path, work_dir, asset_path
DATASET = {dataset!r}
METHOD = {method!r}
WORK_DIR = work_dir(DATASET, METHOD)
DATASET_DIR = PROJECT_ROOT / DATASET
if METHOD == "PrimeKGL":
    SHARED_PRIMEKG.mkdir(parents=True, exist_ok=True)
# A notebook kernel may previously have imported the original project's modules.
for _name in list(sys.modules):
    if _name == "experiments" or _name.startswith("experiments."):
        _file = getattr(sys.modules[_name], "__file__", None)
        if _file and not Path(_file).resolve().is_relative_to(CLINICALKG_ROOT):
            del sys.modules[_name]
if "model1" in sys.modules:
    _file = getattr(sys.modules["model1"], "__file__", None)
    if _file and Path(_file).resolve().parent != WORK_DIR:
        del sys.modules["model1"]
if str(WORK_DIR) in sys.path:
    sys.path.remove(str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR))
if DATASET_DIR.is_dir():
    os.chdir(DATASET_DIR)
os.environ.setdefault("KG_ABLATION_OUTPUT", str(CLINICALKG_ROOT / "results"))
'''


def patch_source(source, dataset, method):
    # Replace path assignments only; preserve function bodies, thresholds, pruning and training logic.
    source = re.sub(r'(?m)^(repo_root|_repo_root|_asset_root|_split_root) = next\(parent.*?\n\s+if .*?\.exists\(\)\)',
                    lambda m: m.group(1) + ' = CLINICALKG_ROOT', source)
    source = re.sub(r'(?m)^(repo_root|_repo_root) = Path\.cwd\(\) if .*? else Path\.cwd\(\)\.parent$',
                    lambda m: m.group(1) + ' = CLINICALKG_ROOT', source)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        value = node.value
        expression = None
        if name == "RAW_KG_PATH":
            expression = "str(raw_primekg_path())"
        elif name in ("ENTITY_EMB_NPY", "ENTITY2ID_FILE", "PATIENT_QUERY_DICT"):
            # Existing calls already use the package-local experiments.config resolver.
            continue
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            filename = value.value
            if filename.endswith((".csv", ".json", ".npy", ".png")):
                if name == "SEED_CSV_PATH" or filename.endswith("_Seed_DementiaHKG.csv"):
                    expression = f"str(WORK_DIR / {filename!r})"
                elif name.endswith("DATA_PATH") or name.endswith("CSV_PATH") and not name.startswith("OUTPUT"):
                    expression = f"str(DATASET_DIR / {filename!r})"
                elif method == "PrimeKGL" and name == "REFINED_KG_PATH":
                    expression = "str(raw_primekg_path())"
                elif method == "PrimeKGL" and name == "OUTPUT_GRAPH_PATH":
                    expression = "str(SHARED_PRIMEKG / 'PrimeKG-G.csv')"
                elif method == "PrimeKGL" and filename.endswith(("LossHistory.json", "LossCurve.png")):
                    expression = f"asset_path(DATASET, 'PrimeKG', {filename!r})"
                elif name.startswith("OUTPUT") or name in ("GRAPH_PATH", "REFINED_KG_PATH", "LOSS_HISTORY_PATH"):
                    expression = f"str(WORK_DIR / {filename!r})"
        if expression is not None:
            replacements.append((node.lineno - 1, node.end_lineno, f"{name} = {expression}\n"))
    for start, end, replacement in reversed(replacements):
        lines[start:end] = [replacement]
    return "".join(lines)


def copy_notebook(source, destination, dataset, method, kind):
    nb = json.loads(source.read_text(encoding="utf-8"))
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        f"# {dataset}_{method}\n",
        "Copied from the original project. Paths are relative to the main folder.\n",
        ("Full PrimeKG embeddings and entity indices are shared by all four cohorts.\n"
         if method == "PrimeKGL" else "Original DemRKG generation and training settings are preserved.\n")]}]
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {"clinicalkg_stage": "bootstrap"}, "outputs": [],
                  "source": bootstrap(dataset, method, kind == "train").splitlines(keepends=True)})
    for index, original in enumerate(nb["cells"]):
        cell = copy.deepcopy(original)
        if cell["cell_type"] == "code":
            source_text = "".join(cell["source"])
            stage = None
            if kind == "build":
                if index == 0:
                    stage = "graph"
                elif "def build_patient_mapping" in source_text:
                    stage = "map"
                elif "class DistMult" in source_text:
                    stage = "kge"
                elif "def plot_loss_curve" in source_text:
                    stage = "plot"
                else:
                    stage = "diagnostic"
            if method == "PrimeKGL" and kind == "build" and stage == "graph":
                source_text = '''# Full PrimeKG is used directly: no seed extraction, blacklist, or Top-k.
RAW_KG_PATH = str(raw_primekg_path())
SHARED_PRIMEKG.mkdir(parents=True, exist_ok=True)
if not Path(RAW_KG_PATH).is_file():
    raise FileNotFoundError(RAW_KG_PATH)
print("Full PrimeKG source:", RAW_KG_PATH)
print("All cohorts share graph-trained assets in:", SHARED_PRIMEKG)
'''
            elif method == "PrimeKGL" and kind == "build" and stage == "kge":
                source_text = '''# Same DistMult score/loss; one shared full-graph pretraining job.
from clinicalKG.full_primekg import train_full_primekg, plot_loss
RESUME = False  # Set True to continue PrimeKG-DistMult-last.pt.
NUM_EPOCHS = 50  # Total target epochs, shared across cohorts.
EMBED_DIM = 128
BATCH_SIZE = 512
LR = 0.005
training_state = train_full_primekg(epochs=NUM_EPOCHS, embed_dim=EMBED_DIM,
                                   batch_size=BATCH_SIZE, lr=LR, resume=RESUME)
plot_loss()
'''
            else:
                source_text = patch_source(source_text, dataset, method)
                if dataset == "NACC" and method == "PrimeKGL" and kind == "train":
                    source_text = source_text.replace("NACC-DementiaHKG", "NACC-PrimeKG")
                    source_text = source_text.replace("'DementiaHKG'", "'PrimeKG'").replace('"DementiaHKG"', '"PrimeKG"')
            cell["source"] = source_text.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            if stage:
                cell.setdefault("metadata", {})["clinicalkg_stage"] = stage
            ast.parse(source_text, filename=str(destination))
        cells.append(cell)
    nb["cells"] = cells
    # Do not carry embedded widget outputs or obsolete run results into the new workflows.
    nb.get("metadata", {}).pop("widgets", None)
    destination.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    record(source, destination, "paths/bootstrap; cleared outputs" +
           ("; Full PrimeKG with shared interruptible DistMult" if method == "PrimeKGL" and kind == "build" else "") +
           ("; NACC PrimeKG reuses original shared architecture template" if dataset == "NACC" and method == "PrimeKGL" and kind == "train" else ""))


def wrapper(destination, notebook, task, method=None):
    target = "build_main" if task == "build" else "train_main"
    args = "Path(__file__).with_name(" + repr(notebook) + ")"
    if method:
        args += ", " + repr(method)
    destination.write_text(f'''"""Command-line entry for {notebook}."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebook_runner import {target}

if __name__ == "__main__":
    {target}({args})
''', encoding="utf-8")


def main():
    support = CLINICAL / "experiments"
    support.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "config.py", "experiment_split.py", "kg_ablation.py", "kg_ablation_stats.py"):
        source, destination = ROOT / "experiments" / name, support / name
        text = source.read_text(encoding="utf-8")
        if name in ("config.py", "experiment_split.py"):
            text = text.replace("ROOT = Path(__file__).resolve().parent.parent",
                                "from clinicalKG.paths import PROJECT_ROOT, work_dir, asset_path\nROOT = PROJECT_ROOT")
        if name == "config.py":
            text = re.sub(r'def resolve_graph_asset\(.*',
                          'def resolve_graph_asset(dataset, graph, filename):\n    return asset_path(dataset, graph, filename)\n',
                          text, flags=re.S)
            start = text.index("def notebook_path(")
            end = text.index("\n\ndef asset_names", start)
            text = text[:start] + '''def notebook_path(dataset, graph):
    method = "PrimeKGL" if graph == "PrimeKG" else "DemRKL"
    name = f"{dataset}三模态+PrimeKG.ipynb" if graph == "PrimeKG" else f"{dataset}_DemRKL.ipynb"
    return work_dir(dataset, method) / name
''' + text[end:]
        destination.write_text(text, encoding="utf-8")
        record(source, destination, "copy; relative main-folder and new asset paths" if name in ("config.py", "experiment_split.py") else "unchanged copy")
    for dataset in DATASETS:
        for method, graph in (("PrimeKGL", "PrimeKG"), ("DemRKL", "DementiaHKG")):
            folder = CLINICAL / method / f"{dataset}_{method}"
            folder.mkdir(parents=True, exist_ok=True)
            # Keep both PrimeKG notebook filenames unchanged as requested.
            build_name = f"Step3-Build-{dataset}-PrimeKG.ipynb" if method == "PrimeKGL" else f"{dataset}_DemRKL_build.ipynb"
            copy_notebook(ROOT / dataset / graph / f"Step3-Build-{dataset}-{graph}.ipynb",
                          folder / build_name, dataset, method, "build")
            train_graph = "DementiaHKG" if dataset == "NACC" and method == "PrimeKGL" else graph
            train_name = f"{dataset}三模态+PrimeKG.ipynb" if method == "PrimeKGL" else f"{dataset}_DemRKL.ipynb"
            copy_notebook(ROOT / dataset / f"{dataset}三模态+{train_graph}.ipynb",
                          folder / train_name, dataset, method, "train")
            for suffix, notebook in (("build", build_name), ("train", train_name)):
                wrapper(folder / f"{dataset}_{method}_{suffix}.py", notebook, suffix,
                        method if suffix == "build" else None)
            source = ROOT / dataset / "model1.py"
            shutil.copyfile(source, folder / source.name)
            record(source, folder / source.name)
            # Curated seed mappings are copied; patient records and historical embeddings are not.
            source = ROOT / dataset / "DementiaHKG" / f"{dataset}_Seed_DementiaHKG.csv"
            shutil.copyfile(source, folder / source.name)
            record(source, folder / source.name)
            if method == "DemRKL":
                for source in sorted((ROOT / dataset / graph).glob("Step[12]*.ipynb")):
                    label = "disease_search" if source.name.startswith("Step1-") else (
                        "unmatched_seed_search" if source.name.startswith("Step2.5-") else "seed_mapping")
                    copy_notebook(source, folder / f"{dataset}_{method}_{label}.ipynb",
                                  dataset, method, "seed")
    (CLINICAL / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Organized {len(manifest)} source copies in {CLINICAL}")


if __name__ == "__main__":
    main()
