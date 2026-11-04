# DemRKGL

See the [run guide](run/README.md) for cohort files, graph asset paths, and training entry points.

See the [DemRKGL workflow](KAMT_DemRKG/README.md) for installation, graph preparation, and cohort-specific training. The [clinicalKG branch](https://github.com/GoqiTea018/DemRKG-model/tree/clinicalKG/clinicalKG) contains the shared comparison workflow.

## Dataset download and organization

The project's MRI dataset is hosted on [Zenodo (record 19646995)](https://zenodo.org/records/19646995), DOI: [10.5281/zenodo.19646995](https://doi.org/10.5281/zenodo.19646995). The record was published on April 19, 2026 and lists the release under **CC BY 4.0**. PrimeKGL and DemRKGL share these cohort inputs.

### Download files

| Cohort | Archive | Size (GB) | NIfTI files | Archive layout |
| --- | --- | ---: | ---: | --- |
| ADNI | [ADNI.zip](https://zenodo.org/records/19646995/files/ADNI.zip?download=1) | 16.77 | 3,146 | `MRI/` |
| AIBL | [AIBL.zip](https://zenodo.org/records/19646995/files/AIBL.zip?download=1) | 3.07 | 516 | `AD/`, `MCI/`, `NC/` |
| PPMI | [PPMI.zip](https://zenodo.org/records/19646995/files/PPMI.zip?download=1) | 3.47 | 414 | `PD/`, `Control/`, `Prodromal/`, `SWEDD/` |
| NACC | [NACC_nii_ad.zip](https://zenodo.org/records/19646995/files/NACC_nii_ad.zip?download=1) | 4.92 | 713 | `NACC_nii_ad/` |
| NACC | [NACC_mci.zip](https://zenodo.org/records/19646995/files/NACC_mci.zip?download=1) | 6.07 | 879 | `NACC_mci/` |
| NACC | [NACC_nii_no.zip](https://zenodo.org/records/19646995/files/NACC_nii_no.zip?download=1) | 14.75 | 2,116 | `NACC_nii_no/` |

Sizes use decimal GB. Counts describe the NIfTI files stored in each archive; training examples are selected by matching the clinical CSV records to MRI files. Zenodo provides an MD5 checksum for each archive.

### Cohorts and clinical tables

Manually download both the [clinical diagnosis CSVs (Zenodo record 23941459)](https://zenodo.org/records/23941459) and the NIfTI MRI archives listed above. Place the CSVs alongside the extracted MRI folders for each cohort.

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

1. Download the archives for the cohorts you plan to use and extract them into a data workspace.
2. Create a cohort directory for each dataset. For AIBL and PPMI, place the extracted class folders under `AIBL/` and `PPMI/`. For NACC, extract all three archives under a common `NACC/` directory.
3. For ADNI, use the MRI identifiers in `AD.csv`, `mci.csv`, and `normal.csv` to organize the files from the archive's `MRI/` directory into `ad_nii_KG/`, `NC_nii_KG/`, and `normal_nii_kg/`, respectively. The configured MCI folder is named `NC_nii_KG/`.
4. Place the clinical CSV tables at the root of each cohort directory and set `CLINICALKG_PROJECT_ROOT` to their common parent. An alternative is to place each cohort directly under `clinicalKG/run/<DATASET>/`.
5. Follow the [DemRKGL guide](KAMT_DemRKG/README.md) to prepare graph assets and run the cohort notebook.

For example, the NACC directory should contain:

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

When using this dataset, cite [Zenodo DOI 10.5281/zenodo.19646995](https://doi.org/10.5281/zenodo.19646995).

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
