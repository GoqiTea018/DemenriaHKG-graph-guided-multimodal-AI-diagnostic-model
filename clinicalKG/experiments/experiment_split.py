"""One fixed patient split for all cohort experiments and clinical norms.

The outer split holds out 20% of patients. The inner split holds out 20%
of the remaining patients for validation: approximately 64/16/20 overall.
Training seeds never enter this module. All visits from a patient stay together.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clinicalKG.paths import PROJECT_ROOT, work_dir, asset_path
ROOT = PROJECT_ROOT
COHORTS = {
    "ADNI": (("AD.csv", 2), ("mci.csv", 1), ("normal.csv", 0)),
    "AIBL": (("AD.csv", 2), ("MCI.csv", 1), ("NC.csv", 0)),
    "NACC": (("NACC_ad.csv", 2), ("NACC_mci.csv", 1), ("NACC_normal.csv", 0)),
    "PPMI": (("PD1.csv", 0), ("control.csv", 1), ("prodromal.csv", 3), ("swedd.csv", 2)),
}


def _identifier(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    # CSV readers can infer the same numeric patient ID as int or float.
    return re.sub(r"^([0-9]+)\.0$", r"\1", value)


def patient_ids(dataset, frame):
    """Recover subject IDs, rather than visit/scan identifiers."""
    result = []
    for _, row in frame.iterrows():
        subject = ""
        columns = {
            "ADNI": ("PTID", "RID"),
            "AIBL": ("Subject", "RID"),
            "NACC": ("NACCID", "ID"),
            "PPMI": ("Subject", "PATNO"),
        }[dataset]
        # Filename prefixes provide a common ID even when another table uses RID.
        name = str(row.get("filename", ""))
        pattern = {"ADNI": r"(?:ADNI_)?(\d{3}_S_\d{4})(?:_|\.)",
                   "AIBL": r"AIBL_(\d+)(?:_|\.)"}.get(dataset)
        match = re.match(pattern, name) if pattern else None
        if match:
            subject = match.group(1)
        else:
            for column in columns:
                value = _identifier(row.get(column, np.nan))
                if value:
                    subject = value
                    break
        if not subject:
            raise ValueError(f"{dataset}: a record lacks a usable patient ID")
        result.append(subject)
    return np.asarray(result)


def cohort_frame(dataset, root=ROOT):
    """Read the common cohort in the order used to concatenate model inputs."""
    frames = []
    for filename, label in COHORTS[dataset]:
        frame = pd.read_csv(Path(root) / dataset / filename)
        # Existing PPMI experiments consistently use the first 120 PD records.
        if dataset == "PPMI" and filename == "PD1.csv":
            frame = frame.iloc[:120].copy()
        frame = frame.copy()
        frame["_label"] = label
        frame["_patient"] = patient_ids(dataset, frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fixed_patient_partition(ids, labels):
    """Stratify unique patients; modal class is used for longitudinal patients.

    Actual record labels are preserved. Sorting makes assignments independent
    of input row order. Ties in the modal class use the smallest class label.
    """
    ids, labels = np.asarray(ids).astype(str), np.asarray(labels).astype(int)
    if len(ids) != len(labels) or not len(ids) or np.any(ids == ""):
        raise ValueError("Patient IDs and labels must be nonempty and aligned")
    subjects = np.unique(ids)
    strata = np.asarray([
        np.unique(labels[ids == subject], return_counts=True)[0][
            np.argmax(np.unique(labels[ids == subject], return_counts=True)[1])]
        for subject in subjects
    ])
    trainval, test, strata_trainval, _ = train_test_split(
        subjects, strata, test_size=0.20, stratify=strata, random_state=32)
    train, val = train_test_split(
        trainval, test_size=0.20, stratify=strata_trainval, random_state=30)
    assignment = {subject: split for split, values in
                  (("Train", train), ("Val", val), ("Test", test)) for subject in values}
    return np.asarray([assignment[subject] for subject in ids])


def cohort_manifest(dataset, root=ROOT):
    frame = cohort_frame(dataset, root)
    frame["_split"] = fixed_patient_partition(frame["_patient"], frame["_label"])
    return frame


def cohort_indices(dataset, labels, root=ROOT):
    frame = cohort_manifest(dataset, root)
    labels = np.asarray(labels)
    if labels.shape != (len(frame),) or not np.array_equal(labels, frame["_label"]):
        raise ValueError(f"{dataset}: input cohort/order differs from the common cohort CSVs")
    indices = tuple(np.flatnonzero(frame["_split"].to_numpy() == split)
                    for split in ("Train", "Val", "Test"))
    expected = set(frame["_label"])
    if any(set(labels[index]) != expected for index in indices):
        raise ValueError(f"{dataset}: a partition is missing a class")
    return indices


def split_cohort_arrays(dataset, labels, *arrays):
    """Return train/val/test for each input array, in that order."""
    indices = cohort_indices(dataset, labels)
    output = []
    for array in arrays:
        if len(array) != len(labels):
            raise ValueError("Feature/knowledge/identity arrays have different lengths")
        output.extend(array[index] for index in indices)
    return output


def training_rows(dataset, frame):
    manifest = cohort_manifest(dataset)
    train_ids = set(manifest.loc[manifest["_split"] == "Train", "_patient"])
    return np.isin(patient_ids(dataset, frame), list(train_ids))


def export_manifest(dataset, destination=None):
    """Export the shared patient list locally (CSV files are Git-ignored)."""
    frame = cohort_manifest(dataset)
    result = frame[["_patient", "_label", "_split"]].copy()
    result.columns = ["PatientID", "Label", "Split"]
    destination = Path(destination) if destination else ROOT / dataset / f"{dataset}_Dataset_Split.csv"
    result.to_csv(destination, index=False)
    return result
