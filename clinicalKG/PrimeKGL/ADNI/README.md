# ADNI step-by-step workflow

In VS Code or Jupyter, open [Run_PrimeKGL.ipynb](../../run/Run_PrimeKGL.ipynb), select the cohort, and run the notebook cells in order.

Follow the [environment setup](../README.md#run-one-cohort-step-by-step) before starting Jupyter. Set CLINICALKG_PROJECT_ROOT to the parent of the four cohort data folders. A local clinicalKG/run/ADNI/ takes precedence over <DATA_ROOT>/ADNI/.

## Inputs

Place ADNI诊断.csv in <DATA_ROOT>/ADNI/ for patient mapping. Put the following classification tables and MRI directories alongside it.

| Clinical CSV | NIfTI directory |
| --- | --- |
| AD.csv | ad_nii_KG/ |
| mci.csv | NC_nii_KG/ |
| normal.csv | normal_nii_kg/ |

## Run the notebooks

1. Run [Full PrimeKG pretraining](../Full_PrimeKG.ipynb) once for all cohorts, or set CLINICALKG_FULL_PRIMEKG_DIR to an existing directory containing PrimeKG-Embeddings.npy, PrimeKG-Entity2ID.json, and PrimeKG-TrainingState.json.
2. Place ADNI_Seed_DementiaHKG.csv in this cohort code folder and run [ADNI patient mapping](Step3-Build-ADNI-PrimeKG.ipynb). The classifier resolves ADNI-PrimeKG-Raw-Entities.json from this code folder or <DATA_ROOT>/ADNI/PrimeKG/.
3. Open the [classification notebook](ADNI三模态%2BPrimeKG.ipynb) in a fresh kernel and execute cells from top to bottom. Its model definitions are in [ADNI_model.py](ADNI_model.py). Check the resolved paths and match the MRI identifiers to the NIfTI filenames before running training.

## Training configuration

Set CLINICALKG_SEED_COUNT to 5 or 10 before opening the kernel, or set KG_ABLATION_SEEDS to a comma-separated list for a custom run. Restart the kernel when changing environment variables. The diagnostic training section uses 200 epochs and a learning rate of 1e-5. The diagnostic DataLoaders use a batch size of 32. The device selection prefers CUDA when available and falls back to CPU. Generated results are written under clinicalKG/analysis/.
