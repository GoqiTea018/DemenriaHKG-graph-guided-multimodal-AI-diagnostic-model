"""Seeded runs; final-epoch validation metrics matching the original notebooks."""

from __future__ import annotations

import csv
import os
import random
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import SEEDS
from .paths import result_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_identity(frames, cohort_names, dataset=None):
    """Make stable record keys and use an explicit subject column when present.

    A filename alone may identify a visit rather than a person. Missing patient
    IDs remain empty so patient-level inference cannot be claimed silently.
    """
    if len(frames) != len(cohort_names):
        raise ValueError("frames and cohort_names must have the same length")
    records, patients = [], []
    for frame, cohort in zip(frames, cohort_names):
        for row_number, (_, row) in enumerate(frame.iterrows()):
            name = str(row.get("filename", row.get("Subject", row.get("PATNO", "")))).strip()
            records.append(f"{cohort}:{row_number}:{name}")
            patient = ""
            for column in (("NACCID", "ID") if dataset == "NACC" else ("NACCID", "PTID", "RID", "Subject", "PATNO")):
                if column in frame.columns and not pd.isna(row[column]):
                    patient = str(row[column]).strip()
                    break
            if not patient and dataset == "ADNI":
                match = re.match(r"^(ADNI_\d{3}_S_\d{4})(?:_|\.)", name)
                patient = match.group(1) if match else ""
            if not patient and dataset == "AIBL":
                match = re.match(r"^(AIBL_\d+)(?:_|\.)", name)
                patient = match.group(1) if match else ""
            patients.append(patient)
    if len(records) != len(set(records)):
        raise ValueError("record keys are not unique")
    if dataset in {"ADNI", "AIBL", "NACC", "PPMI"} and any(not patient for patient in patients):
        raise ValueError(f"Some {dataset} records lack a patient ID")
    return np.asarray(records), np.asarray(patients)


def predict(model, loader, device):
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for features, kg_sequence, kg_mask, target in loader:
            logits = model(features.to(device), kg_sequence.to(device), kg_mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def metrics(labels, probabilities, auc_mode):
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "auc": roc_auc_score(labels, probabilities, multi_class=auc_mode, average="macro"),
    }


def run_ablation(*, dataset, graph, make_model, train_loader, val_loader,
                 test_loader, train_epoch, validate_epoch, epochs, learning_rate,
                 weight_decay, device, test_record_ids, test_patient_ids,
                 auc_mode="ovr", seeds=SEEDS, val_record_ids=None, val_patient_ids=None):
    """Train every epoch, then report validation metrics from the final model.

    The caller supplies a common, fixed validation split and aligned record IDs for
    every graph variant. The output directory must be explicit: predictions
    contain sensitive patient-level information.
    """
    output_root = os.environ.get("KG_ABLATION_OUTPUT")
    if not output_root:
        raise RuntimeError("Set KG_ABLATION_OUTPUT to a private directory before running")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if val_record_ids is None or val_patient_ids is None:
        raise ValueError("Provide val_record_ids and val_patient_ids for final validation metrics")
    if len(val_record_ids) != len(val_loader.dataset):
        raise ValueError("validation record ID count does not match the validation dataset")
    if len(val_patient_ids) != len(val_record_ids):
        raise ValueError("patient ID count does not match record ID count")
    if len(set(val_record_ids)) != len(val_record_ids):
        raise ValueError("validation record IDs must be unique")

    results = []
    last_run = None
    for seed in seeds:
        seed_everything(seed)
        shuffle_generator = torch.Generator().manual_seed(seed)
        seeded_train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            drop_last=train_loader.drop_last,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
            generator=shuffle_generator,
        )
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        history = {f"{prefix}_{key}": [] for prefix in ("train", "val")
                   for key in ("loss", "accuracy", "precision", "recall", "f1", "auc")}
        for epoch in range(1, epochs + 1):
            train_values = train_epoch(model, seeded_train_loader, optimizer, criterion, device)
            val_values = validate_epoch(model, val_loader, criterion, device)
            for prefix, values in (("train", train_values), ("val", val_values)):
                for key, index in (("loss", 0), ("auc", 1), ("f1", 2),
                                   ("recall", 3), ("precision", 4), ("accuracy", 5)):
                    history[f"{prefix}_{key}"].append(float(values[index]) if index < len(values) else float("nan"))
            history_path = result_path(output_dir, dataset, graph, f"history_seed_{seed}.csv")
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("epoch", *history))
                writer.writerows((i + 1, *(history[key][i] for key in history))
                                 for i in range(len(history["train_loss"])))
            print(f"{dataset} {graph} seed={seed} epoch={epoch}/{epochs} "
                  f"train_loss={history['train_loss'][-1]:.4f} "
                  f"train_accuracy={history['train_accuracy'][-1]:.4f} "
                  f"train_precision={history['train_precision'][-1]:.4f} "
                  f"train_recall={history['train_recall'][-1]:.4f} "
                  f"train_auc={history['train_auc'][-1]:.4f} "
                  f"train_f1={history['train_f1'][-1]:.4f} | "
                  f"val_loss={history['val_loss'][-1]:.4f} "
                  f"val_accuracy={history['val_accuracy'][-1]:.4f} "
                  f"val_precision={history['val_precision'][-1]:.4f} "
                  f"val_recall={history['val_recall'][-1]:.4f} "
                  f"val_auc={history['val_auc'][-1]:.4f} "
                  f"val_f1={history['val_f1'][-1]:.4f} "
                  "[Precision/Recall/F1: macro; final epoch model]", flush=True)
        torch.save({"dataset": dataset, "graph": graph, "seed": seed,
                    "epoch": epochs, "model_selection": "final_epoch", "evaluation_split": "val",
                    "model_state_dict": model.state_dict()},
                   result_path(output_dir, dataset, graph, f"last_seed_{seed}.pt"))
        labels, probabilities = predict(model, val_loader, device)
        run_metrics = metrics(labels, probabilities, auc_mode)
        result = {"seed": seed, "epoch": epochs, "evaluation_split": "val", **run_metrics}
        results.append(result)
        path = result_path(output_dir, dataset, graph, f"seed_{seed}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "graph", "seed", "evaluation_split", "record_id", "patient_id",
                             "y_true", "y_pred", *[f"prob_{i}" for i in range(probabilities.shape[1])]))
            for record_id, patient_id, label, probs in zip(val_record_ids,
                                                             val_patient_ids, labels, probabilities):
                writer.writerow((dataset, graph, seed, "val", record_id, patient_id,
                                 int(label), int(np.argmax(probs)), *[float(p) for p in probs]))
        print(f"{dataset} {graph} seed={seed} final_epoch={epochs} val={run_metrics}")
        last_run = {"model": model, "labels": labels, "probabilities": probabilities,
                    "history": history}

    with (result_path(output_dir, dataset, graph, "summary.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return results, last_run
