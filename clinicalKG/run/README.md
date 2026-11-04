# Running clinicalKG experiments

Install the dependencies with `python -m pip install -r clinicalKG/requirements.txt` from the repository root. Start Jupyter after setting `CLINICALKG_PROJECT_ROOT` to the absolute path of your data root. An existing `clinicalKG/run/<DATASET>/` takes precedence over `<DATA_ROOT>/<DATASET>/`.

## Complete runtime layout

Use a repository checkout and a separate data directory. Set `CLINICALKG_PROJECT_ROOT` to the absolute path of `<DATA_ROOT>`. Set `CLINICALKG_RAW_KG` to the absolute path of `kg.csv` when generating graph assets.

```text
<REPO>/
  clinicalKG/
    requirements.txt
    run/
      README.md
      Run_Experiments.ipynb
      common/
    PrimeKGL/                         # Mapping and pretraining notebooks and cohort seed CSVs
    shared/FullPrimeKG/
      PrimeKG-Embeddings.npy
      PrimeKG-Entity2ID.json
      PrimeKG-TrainingState.json
    KAMT_DemRKG/                      # Construction notebooks, seed CSVs, and generated graph CSVs
    analysis/                         # Generated predictions, metrics, and checkpoints

<DATA_ROOT>/
  kg.csv                              # Original PrimeKG for graph generation
  ADNI/
    ADNI诊断.csv
    AD.csv
    ad_nii_KG/
    mci.csv
    NC_nii_KG/
    normal.csv
    normal_nii_kg/
    PrimeKG/
      ADNI-PrimeKG-Raw-Entities.json
    DementiaHKG/
      ADNI-DementiaHKG-Embeddings.npy
      ADNI-DementiaHKG-Entity2ID.json
      ADNI-DementiaHKG-Entities.json
    CustomKG/
      CustomKG-ADNI.npy
      CustomKG-ADNI.txt
  AIBL/
    AIBL诊断.csv
    AD.csv
    AD/
    MCI.csv
    MCI/
    NC.csv
    NC/
    PrimeKG/
      AIBL-PrimeKG-Raw-Entities.json
    DementiaHKG/
      AIBL-DementiaHKG-Embeddings.npy
      AIBL-DementiaHKG-Entity2ID.json
      AIBL-DementiaHKG-Entities.json
    CustomKG/
      CustomKG-AIBL.npy
      CustomKG-AIBL.txt
  NACC/
    NACC诊断.csv
    NACC_ad.csv
    NACC_nii_ad/
    NACC_mci.csv
    NACC_mci/
    NACC_normal.csv
    NACC_nii_no/
    PrimeKG/
      NACC-PrimeKG-Entities.json
    DementiaHKG/
      NACC-DementiaHKG-Embeddings.npy
      NACC-DementiaHKG-Entity2ID.json
      NACC-DementiaHKG-Entities.json
    CustomKG/
      CustomKG-NACC.npy
      CustomKG-NACC.txt
  PPMI/
    PPMI诊断.csv
    PD1.csv
    PD/
    control.csv
    Control/
    prodromal.csv
    Prodromal/
    swedd.csv
    SWEDD/
    PrimeKG/
      PPMI-PrimeKG-Entities.json
    DementiaHKG/
      PPMI-DementiaHKG-Embeddings.npy
      PPMI-DementiaHKG-Entity2ID.json
      PPMI-DementiaHKG-Entities.json
    CustomKG/
      CustomKG-PPMI.npy
      CustomKG-PPMI.txt
```

The NIfTI directories contain the MRI volumes matched by the adjacent classification CSVs. Keep a cohort together in one directory. To use the repository-local layout, place the entire cohort directory under `clinicalKG/run/` with the same contents.

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
- **DemRKGL** ([workflow](../KAMT_DemRKG/README.md)). Place `<DATASET>-DementiaHKG-Embeddings.npy`, `<DATASET>-DementiaHKG-Entity2ID.json`, and `<DATASET>-DementiaHKG-Entities.json` under `<DATA_ROOT>/<DATASET>/DementiaHKG/`. For construction, place `<DATASET>_Seed_DementiaHKG.csv` in `clinicalKG/KAMT_DemRKG/`. The construction notebook writes `<DATASET>_DementiaHKG_PrimeKG_Sub.csv` and `<DATASET>-DementiaHKG-G.csv` there.
- **CustomKG**. Place `CustomKG-<DATASET>.npy` and `CustomKG-<DATASET>.txt` under `<DATA_ROOT>/<DATASET>/CustomKG/`.

## Compare methods

Open [Run_Experiments.ipynb](Run_Experiments.ipynb). Select the cohort and the 5- or 10-seed preset, run `RUN_MODE="check"`, then run `RUN_MODE="all"` to train and compare CustomKG, Full PrimeKG, and DemRKG. Use `RUN_MODE="stats"` for completed runs. The same cohort, seed list, and split must be used across methods.

### Recorded results

The runner writes cohort- and seed-tagged files under `clinicalKG/analysis/`. Per-method filenames also include the graph name.

| Output suffix | Recorded data |
| --- | --- |
| `seed_<seed>.csv` | Evaluation records with record and patient identifiers, true labels, and predictions |
| `history_seed_<seed>.csv` | Epoch-level training and validation history |
| `last_seed_<seed>.pt` | Final model checkpoint |
| `summary.csv` | Per-seed accuracy, macro precision, macro recall, macro F1, macro AUC, evaluation split, and epoch |
| `run_manifest.json` | Evaluation protocol and input/split fingerprints |
| `metrics_mean_sd.csv` | Mean and sample SD of all five metrics across seeds for each method |
| `paired_statistics.csv` | DemRKG versus each baseline on aligned predictions, with metric differences, patient-level bootstrap confidence intervals, permutation p-values, and Holm-adjusted p-values |
| `analysis_definition.json` | Metrics, seeds, resampling settings, and input fingerprints |
| `training_curves.png` | Training curves generated from the saved histories |

The comparison runner reports validation metrics from epoch 200. All four cohorts use macro one-vs-rest AUC.
