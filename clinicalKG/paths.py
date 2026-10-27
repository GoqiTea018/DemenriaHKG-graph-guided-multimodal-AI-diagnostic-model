"""Paths relative to the main project directory; no machine-specific paths."""
from pathlib import Path
import os

CLINICALKG_ROOT = Path(__file__).resolve().parent
# Place clinicalKG/ inside your main folder on either Windows or Linux.
# An optional override points to a different main folder on a cloud server.
PROJECT_ROOT = Path(os.environ.get("CLINICALKG_PROJECT_ROOT", CLINICALKG_ROOT.parent)).expanduser().resolve()
SHARED_PRIMEKG = CLINICALKG_ROOT / "shared" / "FullPrimeKG"
DATASETS = ("ADNI", "AIBL", "NACC", "PPMI")


def work_dir(dataset, method):
    if dataset not in DATASETS or method not in ("PrimeKGL", "DemRKL"):
        raise ValueError((dataset, method))
    return CLINICALKG_ROOT / method / f"{dataset}_{method}"


def raw_primekg_path():
    """Prefer one shared kg.csv; fallbacks reuse the existing local raw file."""
    override = os.environ.get("CLINICALKG_RAW_KG")
    if override:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    candidates = [PROJECT_ROOT / "kg.csv", SHARED_PRIMEKG / "kg.csv"]
    candidates += [PROJECT_ROOT / dataset / kind / "kg.csv"
                   for dataset in DATASETS for kind in ("PrimeKG", "DementiaHKG")]
    return next((path for path in candidates if path.is_file()), candidates[0])


def full_primekg_dir():
    """Use shared outputs or the Full PrimeKG files downloaded into PrimeKGL/."""
    override = os.environ.get("CLINICALKG_FULL_PRIMEKG_DIR")
    if override:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    required = ("PrimeKG-Embeddings.npy", "PrimeKG-Entity2ID.json", "PrimeKG-TrainingState.json")
    return next((path for path in (SHARED_PRIMEKG, CLINICALKG_ROOT / "PrimeKGL")
                 if all((path / name).is_file() for name in required)), SHARED_PRIMEKG)


def _existing_asset(candidates):
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


def asset_path(dataset, graph, filename):
    """Keep legacy callers, sharing only Full PrimeKG's graph-trained assets."""
    if graph == "PrimeKG":
        for suffix in ("Embeddings.npy", "Entity2ID.json", "Relation2ID.json",
                       "LossHistory.json", "LossCurve.png"):
            if filename.endswith("-" + suffix):
                return str(full_primekg_dir() / ("PrimeKG-" + suffix))
        return _existing_asset((work_dir(dataset, "PrimeKGL") / filename,
                                PROJECT_ROOT / dataset / "PrimeKG" / filename,
                                PROJECT_ROOT / dataset / filename))
    if graph == "DementiaHKG":
        return _existing_asset((work_dir(dataset, "DemRKL") / filename,
                                PROJECT_ROOT / dataset / "DementiaHKG" / filename,
                                PROJECT_ROOT / dataset / filename))
    if graph == "CustomKG":
        return _existing_asset((PROJECT_ROOT / dataset / "CustomKG" / filename,
                                PROJECT_ROOT / dataset / filename))
    raise ValueError(f"This package contains PrimeKG and DemRKG, received {graph}")
