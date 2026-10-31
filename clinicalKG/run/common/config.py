"""Locations shared by notebook and command-line entry points."""

from pathlib import Path

from clinicalKG.run.common.paths import PROJECT_ROOT, work_dir, asset_path
ROOT = PROJECT_ROOT
SEEDS = (11, 23, 37, 51, 67)
DATASETS = ("ADNI", "AIBL", "NACC", "PPMI")
GRAPHS = ("CustomKG", "PrimeKG", "DementiaHKG")
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
