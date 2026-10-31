"""Central paths for portable notebooks and local-only assets."""
from pathlib import Path
import os

CLINICALKG_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = CLINICALKG_ROOT / "run"
ANALYSIS_ROOT = CLINICALKG_ROOT / "analysis"
LOCAL_PROJECT_ROOT = Path(r"C:\Users\HP\Design\DemenriaHKG graph-guided multimodal AI diagnostic model")
PROJECT_ROOT = Path(os.environ.get("CLINICALKG_PROJECT_ROOT", str(LOCAL_PROJECT_ROOT))).expanduser().resolve()
SHARED_PRIMEKG = CLINICALKG_ROOT / "shared" / "FullPrimeKG"
DATASETS = ("ADNI", "AIBL", "NACC", "PPMI")

def dataset_dir(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    local = RUN_ROOT / dataset
    return local if local.is_dir() else PROJECT_ROOT / dataset

def work_dir(dataset, method):
    if dataset not in DATASETS or method not in ("PrimeKGL", "KAMT_DemRKG", "DemRKL"):
        raise ValueError((dataset, method))
    return CLINICALKG_ROOT / ("KAMT_DemRKG" if method == "DemRKL" else method)

def raw_primekg_path():
    override = os.environ.get("CLINICALKG_RAW_KG")
    if override:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    candidates = [RUN_ROOT / "kg.csv", PROJECT_ROOT / "kg.csv", SHARED_PRIMEKG / "kg.csv"]
    candidates += [dataset_dir(d) / kind / "kg.csv" for d in DATASETS for kind in ("PrimeKG", "DementiaHKG")]
    return next((p for p in candidates if p.is_file()), candidates[0])

def full_primekg_dir():
    override = os.environ.get("CLINICALKG_FULL_PRIMEKG_DIR")
    if override:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    required = ("PrimeKG-Embeddings.npy", "PrimeKG-Entity2ID.json", "PrimeKG-TrainingState.json")
    return next((p for p in (SHARED_PRIMEKG, CLINICALKG_ROOT / "PrimeKGL")
                 if all((p / n).is_file() for n in required)), SHARED_PRIMEKG)

def asset_path(dataset, graph, filename):
    if graph == "PrimeKG":
        for suffix in ("Embeddings.npy", "Entity2ID.json", "Relation2ID.json", "LossHistory.json", "LossCurve.png"):
            if filename.endswith("-" + suffix):
                return str(full_primekg_dir() / ("PrimeKG-" + suffix))
        method = "PrimeKGL"
    elif graph == "DementiaHKG":
        method = "KAMT_DemRKG"
    elif graph == "CustomKG":
        method = None
    else:
        raise ValueError(graph)
    candidates = ([work_dir(dataset, method) / filename] if method else [])
    candidates += [dataset_dir(dataset) / graph / filename, dataset_dir(dataset) / filename]
    return str(next((p for p in candidates if p.is_file()), candidates[0]))

def result_path(root, dataset, graph, filename):
    """Flat filenames include the exact seed set to keep experiments separate."""
    from .config import EXPERIMENT_TAG
    if dataset not in DATASETS or graph not in (None, "CustomKG", "PrimeKG", "DementiaHKG"):
        raise ValueError((dataset, graph))
    filename = str(filename)
    if Path(filename).name != filename:
        raise ValueError(filename)
    prefix = dataset + "_" + EXPERIMENT_TAG + ("_" + graph if graph else "")
    return Path(root) / f"{prefix}_{filename}"
