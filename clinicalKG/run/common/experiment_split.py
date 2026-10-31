"""Original record-level stratified splits shared across KG variants and seeds."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clinicalKG.run.common.paths import PROJECT_ROOT, work_dir, asset_path, dataset_dir, result_path, ANALYSIS_ROOT
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


def cohort_frame(dataset, root=None):
    """Read the common cohort in the order used to concatenate model inputs."""
    frames = []
    for filename, label in COHORTS[dataset]:
        frame = pd.read_csv((Path(root) / dataset if root is not None else dataset_dir(dataset)) / filename)
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


def original_record_indices(labels):
    """Reproduce the original notebooks, including shuffled record order."""
    labels = np.asarray(labels)
    trainval, test, trainval_labels, _ = train_test_split(
        np.arange(len(labels)), labels, test_size=0.20,
        stratify=labels, random_state=32)
    train, val = train_test_split(
        trainval, test_size=0.20, stratify=trainval_labels, random_state=30)
    return train, val, test


def cohort_manifest(dataset, root=None):
    frame = cohort_frame(dataset, root)
    frame["_split"] = ""
    for name, indices in zip(("Train", "Val", "Test"), original_record_indices(frame["_label"])):
        frame.loc[indices, "_split"] = name
    return frame


def cohort_indices(dataset, labels, root=None):
    frame = cohort_manifest(dataset, root)
    labels = np.asarray(labels)
    if labels.shape != (len(frame),) or not np.array_equal(labels, frame["_label"]):
        raise ValueError(f"{dataset}: input cohort/order differs from the common cohort CSVs")
    indices = original_record_indices(labels)
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
    destination = Path(destination) if destination else result_path(ANALYSIS_ROOT, dataset, None, "Dataset_Split.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
