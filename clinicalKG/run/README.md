# Running clinicalKG experiments

Install the dependencies with `python -m pip install -r clinicalKG/requirements.txt` from the repository root. Start Jupyter after setting `CLINICALKG_PROJECT_ROOT` to the absolute path of your data root. An existing `clinicalKG/run/<DATASET>/` takes precedence over `<DATA_ROOT>/<DATASET>/`.

## Cohort inputs

Manually download [clinical CSVs](https://zenodo.org/records/23941459) and [NIfTI MRI archives](https://zenodo.org/records/19646995). Place all listed CSVs and MRI directories directly under the indicated cohort directory. Each diagnosis CSV is the consolidated input for clinical-to-graph mapping. The classification notebooks read the class-specific CSVs and match their MRI identifiers to the corresponding NIfTI directories.

| Cohort directory | Diagnosis CSV | Classification CSV → NIfTI directory |
| --- | --- | --- |
| ADNI | `ADNI诊断.csv` | `AD.csv` → `ad_nii_KG/`, `mci.csv` → `NC_nii_KG/`, `normal.csv` → `normal_nii_kg/` |
| AIBL | `AIBL诊断.csv` | `AD.csv` → `AD/`, `MCI.csv` → `MCI/`, `NC.csv` → `NC/` |
| NACC | `NACC诊断.csv` | `NACC_ad.csv` → `NACC_nii_ad/`, `NACC_mci.csv` → `NACC_mci/`, `NACC_normal.csv` → `NACC_nii_no/` |
| PPMI | `PPMI诊断.csv` | `PD1.csv` → `PD/`, `control.csv` → `Control/`, `prodromal.csv` → `Prodromal/`, `swedd.csv` → `SWEDD/` |

For ADNI, organize the extracted `MRI/` files into the three folders using the MRI identifiers in the classification CSVs. Extract all three NACC archives into the same cohort directory.

## Graph inputs

To generate graph assets from scratch, set `CLINICALKG_RAW_KG` to the original PrimeKG `kg.csv`. Use the method guides below for graph generation and clinical mapping. Keep filenames and capitalization as listed.

- **DemRKGL** ([workflow](../KAMT_DemRKG/README.md)). Place `<DATASET>-DementiaHKG-Embeddings.npy`, `<DATASET>-DementiaHKG-Entity2ID.json`, and `<DATASET>-DementiaHKG-Entities.json` under `<DATA_ROOT>/<DATASET>/DementiaHKG/`. For construction, place `<DATASET>_Seed_DementiaHKG.csv` in `clinicalKG/KAMT_DemRKG/`. The construction notebook writes `<DATASET>_DementiaHKG_PrimeKG_Sub.csv` and `<DATASET>-DementiaHKG-G.csv` there.

## Train DemRKGL

Open [Run_DemRKGL.ipynb](Run_DemRKGL.ipynb), select a cohort and run the input check before enabling training. For cross-method comparisons, use the [clinicalKG comparison runner and output guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/run/README.md).
