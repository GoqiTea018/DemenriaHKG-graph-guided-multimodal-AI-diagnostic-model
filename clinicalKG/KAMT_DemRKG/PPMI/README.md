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

1. Set CLINICALKG_RAW_KG to the original PrimeKG kg.csv. Open [PPMI disease search](PPMI_KAMT_DemRKG_disease_search.ipynb) and execute its cells in order to identify candidate entities.
2. Open [PPMI seed mapping](PPMI_KAMT_DemRKG_seed_mapping.ipynb) to align the clinical fields and prepare PPMI_Seed_DementiaHKG.csv in this cohort code directory.
3. Run [PPMI graph construction](PPMI_KAMT_DemRKG_build.ipynb) to build the graph, create the patient mapping, and train DistMult. The classifier reads PPMI-DementiaHKG-Embeddings.npy, PPMI-DementiaHKG-Entity2ID.json, and PPMI-DementiaHKG-Entities.json from this code folder or <DATA_ROOT>/PPMI/DementiaHKG/.
4. Open the [classification notebook](PPMI三模态%2BDemRKG.ipynb) in a fresh kernel and execute cells from top to bottom. Its model definitions are in [PPMI_model.py](PPMI_model.py). Check the resolved paths and match the MRI identifiers to the NIfTI filenames before running training.

## Training configuration

Set CLINICALKG_SEED_COUNT to 5 or 10 before opening the kernel, or set KG_ABLATION_SEEDS to a comma-separated list for a custom run. Restart the kernel when changing environment variables. The diagnostic training section uses 200 epochs and a learning rate of 1e-5. The diagnostic DataLoaders use a batch size of 32. The device selection prefers CUDA when available and falls back to CPU. Generated results are written under clinicalKG/analysis/.
