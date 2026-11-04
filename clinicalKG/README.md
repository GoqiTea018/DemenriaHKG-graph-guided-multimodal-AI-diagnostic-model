# DemRKGL

See the [run guide](run/README.md) for the [complete runtime layout](run/README.md#complete-runtime-layout), cohort files, and training entry points.

See the [DemRKGL workflow](KAMT_DemRKG/README.md) for installation, graph preparation, and cohort-specific training. The [clinicalKG branch](https://github.com/GoqiTea018/DemRKG-model/tree/clinicalKG/clinicalKG) contains the shared comparison workflow.

## Dataset organization

### Cohorts and clinical tables

Manually prepare the cohort clinical CSVs and NIfTI MRI files. Place the CSVs alongside the extracted MRI folders for each cohort.

The consolidated diagnosis tables (`ADNI诊断.csv`, `AIBL诊断.csv`, `NACC诊断.csv`, `PPMI诊断.csv`) are inputs to the DemRKG construction notebooks. The notebooks use the clinical records to build patient-to-graph mappings that associate each record with relevant DemRKG entities.

The class-specific CSVs listed below contain the same records grouped by diagnostic label. The downstream classification notebooks combine samples from those tables and apply the cohort-specific training, validation, and test split protocol.

| Cohort | Diagnostic groups | Clinical CSV -> MRI folder |
| --- | --- | --- |
| ADNI | Alzheimer's disease, mild cognitive impairment, normal controls | `AD.csv` -> `ad_nii_KG/`; `mci.csv` -> `NC_nii_KG/`; `normal.csv` -> `normal_nii_kg/` |
| AIBL | Alzheimer's disease, mild cognitive impairment, normal controls | `AD.csv` -> `AD/`; `MCI.csv` -> `MCI/`; `NC.csv` -> `NC/` |
| NACC | Alzheimer's disease, mild cognitive impairment, normal controls | `NACC_ad.csv` -> `NACC_nii_ad/`; `NACC_mci.csv` -> `NACC_mci/`; `NACC_normal.csv` -> `NACC_nii_no/` |
| PPMI | Parkinson's disease, controls, prodromal participants, SWEDD | `PD1.csv` -> `PD/`; `control.csv` -> `Control/`; `prodromal.csv` -> `Prodromal/`; `swedd.csv` -> `SWEDD/` |

SWEDD denotes scans without evidence of dopaminergic deficit. The clinical tables provide the cohort-specific tabular inputs and MRI identifiers consumed by the training notebooks. Their required names and MRI folder pairs are defined in [run/common/config.py](run/common/config.py).

### Prepare the directory layout

1. Prepare the clinical CSVs and NIfTI volumes for each cohort in a local data workspace.
2. Create a cohort directory for each dataset. For AIBL and PPMI, place the extracted class folders under `AIBL/` and `PPMI/`. For NACC, extract all three archives under a common `NACC/` directory.
3. For ADNI, use the MRI identifiers in `AD.csv`, `mci.csv`, and `normal.csv` to organize the files from the archive's `MRI/` directory into `ad_nii_KG/`, `NC_nii_KG/`, and `normal_nii_kg/`, respectively. The configured MCI folder is named `NC_nii_KG/`.
4. Place the clinical CSV tables at the root of each cohort directory and set `CLINICALKG_PROJECT_ROOT` to their common parent. An alternative is to place each cohort directly under `clinicalKG/run/<DATASET>/`.
5. Follow the [DemRKGL guide](KAMT_DemRKG/README.md) to prepare graph assets and run the cohort notebook.

The NACC directory layout is shown below.

```text
<DATA_ROOT>/
`-- NACC/
    |-- NACC_ad.csv
    |-- NACC_mci.csv
    |-- NACC_normal.csv
    |-- NACC_nii_ad/
    |-- NACC_mci/
    `-- NACC_nii_no/
```

Follow the [DemRKGL guide](KAMT_DemRKG/README.md) to configure the original PrimeKG and prepare graph inputs.

## Data and graph paths

Paths are centralized in [run/common/paths.py](run/common/paths.py). Place each cohort's clinical CSV files, MRI folders, and graph assets under `clinicalKG/run/<DATASET>/`, or set `CLINICALKG_PROJECT_ROOT` to the parent directory containing `ADNI/`, `AIBL/`, `NACC/`, and `PPMI/` before starting Jupyter.

| Environment variable | Purpose |
| --- | --- |
| `CLINICALKG_PROJECT_ROOT` | Parent directory of the four cohort data directories |
| `CLINICALKG_RAW_KG` | Path to the complete raw PrimeKG `kg.csv` |

For example, in a Bash session:

```bash
export CLINICALKG_PROJECT_ROOT=/path/to/cohort-data
export CLINICALKG_RAW_KG=/path/to/PrimeKG/kg.csv
python -m jupyterlab
```

In PowerShell, use `$env:CLINICALKG_PROJECT_ROOT = "D:/data"` and `$env:CLINICALKG_RAW_KG = "D:/data/kg.csv"`.

The `run/<DATASET>/` directory takes precedence over the corresponding directory under `CLINICALKG_PROJECT_ROOT`. Cohort-specific CSV and MRI folder names are defined in [run/common/config.py](run/common/config.py). Graph assets can reside in the method directory or in the cohort's `DementiaHKG/` directory.
