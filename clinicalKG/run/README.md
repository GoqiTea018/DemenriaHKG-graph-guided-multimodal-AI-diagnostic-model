# Running clinicalKG experiments

Install the dependencies with `python -m pip install -r clinicalKG/requirements.txt` from the repository root. Start Jupyter after setting `CLINICALKG_PROJECT_ROOT` to the absolute path of your data root. An existing `clinicalKG/run/<DATASET>/` takes precedence over `<DATA_ROOT>/<DATASET>/`.

## Cohort inputs

Manually prepare the cohort clinical CSVs and NIfTI MRI files. Place all listed CSVs and MRI directories directly under the indicated cohort directory. Each diagnosis CSV is the consolidated input for clinical-to-graph mapping. The classification notebooks read the class-specific CSVs and match their MRI identifiers to the corresponding NIfTI directories.

| Cohort directory | Diagnosis CSV | Classification CSV → NIfTI directory |
| --- | --- | --- |
| ADNI | `ADNI诊断.csv` | `AD.csv` → `ad_nii_KG/`, `mci.csv` → `NC_nii_KG/`, `normal.csv` → `normal_nii_kg/` |
| AIBL | `AIBL诊断.csv` | `AD.csv` → `AD/`, `MCI.csv` → `MCI/`, `NC.csv` → `NC/` |
| NACC | `NACC诊断.csv` | `NACC_ad.csv` → `NACC_nii_ad/`, `NACC_mci.csv` → `NACC_mci/`, `NACC_normal.csv` → `NACC_nii_no/` |
| PPMI | `PPMI诊断.csv` | `PD1.csv` → `PD/`, `control.csv` → `Control/`, `prodromal.csv` → `Prodromal/`, `swedd.csv` → `SWEDD/` |

For ADNI, organize the extracted `MRI/` files into the three folders using the MRI identifiers in the classification CSVs. Place all three NACC MRI folders in the same cohort directory.

## Graph inputs

To generate graph assets from scratch, set `CLINICALKG_RAW_KG` to the original PrimeKG `kg.csv`. Use the method guides below for graph generation and clinical mapping. Keep filenames and capitalization as listed.

- **PrimeKGL** ([workflow](../PrimeKGL/README.md)). Place `PrimeKG-Embeddings.npy`, `PrimeKG-Entity2ID.json`, and `PrimeKG-TrainingState.json` together in `clinicalKG/shared/FullPrimeKG/` or set `CLINICALKG_FULL_PRIMEKG_DIR` to their directory. Copy each patient mapping into `<DATA_ROOT>/<DATASET>/PrimeKG/`: `ADNI-PrimeKG-Raw-Entities.json`, `AIBL-PrimeKG-Raw-Entities.json`, `NACC-PrimeKG-Entities.json`, or `PPMI-PrimeKG-Entities.json`. Seed CSVs used for mapping go in `clinicalKG/PrimeKGL/` as `<DATASET>_Seed_DementiaHKG.csv`.

## Train PrimeKGL

Open [Run_PrimeKGL.ipynb](Run_PrimeKGL.ipynb), select a cohort and run the input check before enabling training. The epoch-200 validation report includes accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest AUC. For cross-method comparisons, use the [clinicalKG comparison runner and output guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/run/README.md).
