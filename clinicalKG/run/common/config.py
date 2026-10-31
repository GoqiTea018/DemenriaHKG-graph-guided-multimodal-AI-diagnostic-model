"""Locations shared by notebook and command-line entry points."""

from pathlib import Path
import os

from clinicalKG.run.common.paths import PROJECT_ROOT, work_dir, asset_path
ROOT = PROJECT_ROOT
DEFAULT_SEEDS = (51, 37, 67, 71, 109, 27, 11, 23, 97, 83)
_seed_override = os.environ.get("KG_ABLATION_SEEDS")
SEEDS = tuple(int(value.strip()) for value in _seed_override.split(",")) if _seed_override else DEFAULT_SEEDS
if not SEEDS or len(SEEDS) != len(set(SEEDS)) or any(seed < 0 or seed >= 2**32 for seed in SEEDS):
    raise ValueError("KG_ABLATION_SEEDS must contain distinct integers in [0, 2**32)")
EXPERIMENT_TAG = "seeds_" + "-".join(map(str, SEEDS))
DATASETS = ("ADNI", "AIBL", "NACC", "PPMI")
GRAPHS = ("CustomKG", "PrimeKG", "DementiaHKG")


def auc_mode(dataset):
    """Use the same multi-class AUC definition for epoch logs and final scores."""
    if dataset not in DATASETS:
        raise ValueError(dataset)
    return "ovo" if dataset == "AIBL" else "ovr"


MRI = {
    "ADNI": (("AD.csv", "ad_nii_KG"), ("mci.csv", "NC_nii_KG"), ("normal.csv", "normal_nii_kg")),
    "AIBL": (("AD.csv", "AD"), ("MCI.csv", "MCI"), ("NC.csv", "NC")),
    "NACC": (("NACC_ad.csv", "NACC_nii_ad"), ("NACC_mci.csv", "NACC_mci"), ("NACC_normal.csv", "NACC_nii_no")),
    "PPMI": (("PD1.csv", "PD"), ("control.csv", "Control"), ("prodromal.csv", "Prodromal"), ("swedd.csv", "SWEDD")),
}


def notebook_path(dataset, graph):
    from .paths import RUN_ROOT
    if graph == "CustomKG":
        return RUN_ROOT / f"{dataset}三模态+CustomKG.ipynb"
    method = "PrimeKGL" if graph == "PrimeKG" else "KAMT_DemRKG"
    kg_name = "DemRKG" if graph == "DementiaHKG" else graph
    return work_dir(dataset, method) / f"{dataset}三模态+{kg_name}.ipynb"


def model_path(dataset, graph):
    if graph == "CustomKG":
        return Path(__file__).with_name(f"{dataset}_customkg_model.py")
    return work_dir(dataset, "PrimeKGL" if graph == "PrimeKG" else "KAMT_DemRKG") / f"{dataset}_model.py"


def asset_names(dataset, graph):
    if graph == "CustomKG":
        return {"ENTITY_EMB_NPY": f"CustomKG-{dataset}.npy",
                "ENTITY2ID_FILE": f"CustomKG-{dataset}.txt"}
    prefix = f"{dataset}-{graph}"
    if graph == "PrimeKG" and dataset in ("ADNI", "AIBL"):
        prefix += "-Raw"
    return {"ENTITY_EMB_NPY": prefix + "-Embeddings.npy",
            "ENTITY2ID_FILE": prefix + "-Entity2ID.json",
            "PATIENT_QUERY_DICT": prefix + "-Entities.json"}


def resolve_graph_asset(dataset, graph, filename):
    return asset_path(dataset, graph, filename)
