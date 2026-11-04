# PPMI step-by-step workflow

Follow the [environment setup](../README.md#run-one-cohort-step-by-step) before starting Jupyter. Set CLINICALKG_PROJECT_ROOT to the parent of the four cohort data folders. A local clinicalKG/run/PPMI/ takes precedence over <DATA_ROOT>/PPMI/.

## Inputs

Place PPMI诊断.csv in <DATA_ROOT>/PPMI/ for patient mapping. Put the following classification tables and MRI directories alongside it.

| Clinical CSV | NIfTI directory |
| --- | --- |
| PD1.csv | PD/ |
| control.csv | Control/ |
| prodromal.csv | Prodromal/ |
| swedd.csv | SWEDD/ |

## Run the notebooks

1. Run [Full PrimeKG pretraining](../Full_PrimeKG.ipynb) once for all cohorts, or set CLINICALKG_FULL_PRIMEKG_DIR to an existing directory containing PrimeKG-Embeddings.npy, PrimeKG-Entity2ID.json, and PrimeKG-TrainingState.json.
2. Place PPMI_Seed_DementiaHKG.csv in this cohort code folder and run [PPMI patient mapping](Step3-Build-PPMI-PrimeKG.ipynb). The classifier resolves PPMI-PrimeKG-Entities.json from this code folder or <DATA_ROOT>/PPMI/PrimeKG/.
3. Open the [classification notebook](PPMI三模态%2BPrimeKG.ipynb) in a fresh kernel and execute cells from top to bottom. Its model definitions are in [PPMI_model.py](PPMI_model.py). Check the resolved paths and match the MRI identifiers to the NIfTI filenames before running training.

## Training configuration

Set CLINICALKG_SEED_COUNT to 5 or 10 before opening the kernel, or set KG_ABLATION_SEEDS to a comma-separated list for a custom run. Restart the kernel when changing environment variables. The diagnostic training section uses 200 epochs and a learning rate of 1e-5. The diagnostic DataLoaders use a batch size of 32. The device selection prefers CUDA when available and falls back to CPU. Generated results are written under clinicalKG/analysis/.
