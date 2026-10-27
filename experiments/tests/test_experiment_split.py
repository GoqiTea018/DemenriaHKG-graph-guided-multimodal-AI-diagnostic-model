import numpy as np
import pandas as pd
import pytest

from experiments.experiment_split import fixed_patient_partition, patient_ids


def test_repeated_visits_stay_together_and_row_order_does_not_change_split():
    subjects = np.asarray([f"subject_{i:03d}" for i in range(120)])
    ids = np.repeat(subjects, 2)
    labels = np.repeat(np.arange(120) % 3, 2)
    # A longitudinal patient changes class; both visits still stay together.
    labels[1] = 1
    original = fixed_patient_partition(ids, labels)
    for subject in subjects:
        assert len(set(original[ids == subject])) == 1
    permutation = np.random.default_rng(87).permutation(len(ids))
    shuffled = fixed_patient_partition(ids[permutation], labels[permutation])
    assert np.array_equal(shuffled, original[permutation])
    assert set(original) == {"Train", "Val", "Test"}
    assert all(set(labels[original == split]) == {0, 1, 2}
               for split in ("Train", "Val", "Test"))


def test_training_seed_does_not_change_patient_partition():
    ids = np.asarray([f"p{i}" for i in range(150)])
    labels = np.arange(150) % 3
    np.random.seed(11)
    first = fixed_patient_partition(ids, labels)
    np.random.seed(67)
    assert np.array_equal(first, fixed_patient_partition(ids, labels))


def test_patient_ids_use_subject_prefix_and_normalize_numeric_ids():
    adni = pd.DataFrame({"filename": ["ADNI_001_S_1234_visit1.nii", "ADNI_001_S_1234_visit2.nii"]})
    assert patient_ids("ADNI", adni).tolist() == ["001_S_1234"] * 2
    ppmi = pd.DataFrame({"Subject": [1234.0, 1234]})
    assert patient_ids("PPMI", ppmi).tolist() == ["1234"] * 2
    with pytest.raises(ValueError, match="patient ID"):
        patient_ids("PPMI", pd.DataFrame({"filename": ["unknown.nii"]}))
