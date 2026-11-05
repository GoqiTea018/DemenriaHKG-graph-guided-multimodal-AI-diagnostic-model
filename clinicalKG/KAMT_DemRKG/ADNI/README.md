# ADNI step-by-step workflow

In VS Code or Jupyter, open [Run_DemRKGL.ipynb](../../run/Run_DemRKGL.ipynb), select the cohort, and run the notebook cells in order.

Follow the [environment setup](../README.md#run-one-cohort-step-by-step) before starting Jupyter. Set CLINICALKG_PROJECT_ROOT to the parent of the four cohort data folders. A local clinicalKG/run/ADNI/ takes precedence over <DATA_ROOT>/ADNI/.

## Inputs

Place ADNI诊断.csv in <DATA_ROOT>/ADNI/ for patient mapping. Put the following classification tables and MRI directories alongside it.

| Clinical CSV | NIfTI directory |
| --- | --- |
| AD.csv | ad_nii_KG/ |
| mci.csv | NC_nii_KG/ |
| normal.csv | normal_nii_kg/ |

## Run the notebooks

1. Set CLINICALKG_RAW_KG to the original PrimeKG kg.csv. Open [ADNI disease search](ADNI_KAMT_DemRKG_disease_search.ipynb) and execute its cells in order to identify candidate entities.
2. Open [ADNI seed mapping](ADNI_KAMT_DemRKG_seed_mapping.ipynb) to align the clinical fields and prepare ADNI_Seed_DementiaHKG.csv in this cohort code directory.
3. Run [ADNI graph construction](ADNI_KAMT_DemRKG_build.ipynb) to build the graph, create the patient mapping, and train DistMult. The classifier reads ADNI-DementiaHKG-Embeddings.npy, ADNI-DementiaHKG-Entity2ID.json, and ADNI-DementiaHKG-Entities.json from this code folder or <DATA_ROOT>/ADNI/DementiaHKG/.
4. Open the [classification notebook](ADNI三模态%2BDemRKG.ipynb) in a fresh kernel and execute cells from top to bottom. Its model definitions are in [ADNI_model.py](ADNI_model.py). Check the resolved paths and match the MRI identifiers to the NIfTI filenames before running training.

## Training configuration

Set CLINICALKG_SEED_COUNT to 5 or 10 before opening the kernel, or set KG_ABLATION_SEEDS to a comma-separated list for a custom run. Restart the kernel when changing environment variables. The classifier defaults to 200 epochs, a batch size of 32, and a learning rate of 1e-5. The notebook uses CUDA when available and otherwise selects CPU. Generated results are written under clinicalKG/analysis/.
