# Running clinicalKG experiments

In VS Code or Jupyter, open [Run_PrimeKGL.ipynb](Run_PrimeKGL.ipynb), select the cohort, and run the notebook cells in order.

For graph preparation and cell-by-cell training, start from the [four cohort guides](../PrimeKGL/README.md#run-one-cohort-step-by-step).

Use a kernel with the dependencies in `clinicalKG/requirements.txt`. Start Jupyter after setting `CLINICALKG_PROJECT_ROOT` to the absolute path of your data root. An existing `clinicalKG/run/<DATASET>/` takes precedence over `<DATA_ROOT>/<DATASET>/`.

## Required downloads

| Download | Contents | Destination |
| --- | --- | --- |
| Clinical tables | Consolidated diagnosis CSV and class-specific CSVs for the selected cohort | Place the CSVs directly in `<DATA_ROOT>/<DATASET>/`. |
| MRI data | NIfTI volumes for the selected cohort | Extract or organize the volumes into the named MRI folders under `<DATA_ROOT>/<DATASET>/`, as shown below. |
| Graph assets | Precomputed embeddings, indices, and cohort mappings required by the selected workflow | Follow the graph paths in the directory tree and Graph inputs section. |

For graph construction, obtain the original PrimeKG `kg.csv` and set `CLINICALKG_RAW_KG` to its path.

## Complete runtime layout

Use a repository checkout and a separate data directory. Set `CLINICALKG_PROJECT_ROOT` to the absolute path of `<DATA_ROOT>`. Set `CLINICALKG_RAW_KG` to the absolute path of `kg.csv` when generating graph assets.

```text
<REPO>/
  clinicalKG/
    requirements.txt
    run/
      README.md
      Run_PrimeKGL.ipynb
      common/
    PrimeKGL/                         # Mapping and pretraining notebooks and cohort seed CSVs
    shared/FullPrimeKG/
      PrimeKG-Embeddings.npy
      PrimeKG-Entity2ID.json
      PrimeKG-TrainingState.json
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

- **PrimeKGL** ([workflow](../PrimeKGL/README.md)). Place `PrimeKG-Embeddings.npy`, `PrimeKG-Entity2ID.json`, and `PrimeKG-TrainingState.json` together in `clinicalKG/shared/FullPrimeKG/` or set `CLINICALKG_FULL_PRIMEKG_DIR` to their directory. Copy each patient mapping into `<DATA_ROOT>/<DATASET>/PrimeKG/`: `ADNI-PrimeKG-Raw-Entities.json`, `AIBL-PrimeKG-Raw-Entities.json`, `NACC-PrimeKG-Entities.json`, or `PPMI-PrimeKG-Entities.json`. Seed CSVs used for mapping go in `clinicalKG/PrimeKGL/<DATASET>/` as `<DATASET>_Seed_DementiaHKG.csv`.

## Train PrimeKGL

Open [Run_PrimeKGL.ipynb](Run_PrimeKGL.ipynb), select a cohort and run the input check before enabling training. The epoch-200 validation report includes accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest AUC. For cross-method comparisons, use the [clinicalKG comparison runner and output guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/run/README.md).

## Compare DemRKG and Full PrimeKG

Use the [clinicalKG comparison guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/README.md) to run both methods under the shared protocol. The comparison reports accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest ROC AUC on the epoch-200 validation split. The paired analysis reports metric differences on aligned predictions, bootstrap confidence intervals, permutation p-values, and Holm-adjusted p-values.
