"""Locations shared by notebook and command-line entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    # NACC uses one architecture template for both external graph sources.
    template = "DementiaHKG" if dataset == "NACC" and graph == "PrimeKG" else graph
    return ROOT / dataset / f"{dataset}三模态+{template}.ipynb"


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
    """Prefer graph-building outputs to older copies in the cohort directory."""
    candidates = (ROOT / dataset / graph / filename, ROOT / dataset / filename)
    return str(next((path for path in candidates if path.is_file()), candidates[0]))
