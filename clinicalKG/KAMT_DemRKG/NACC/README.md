# NACC step-by-step workflow

In VS Code or Jupyter, open [Run_DemRKGL.ipynb](../../run/Run_DemRKGL.ipynb), select the cohort, and run the notebook cells in order.

Follow the [environment setup](../README.md#run-one-cohort-step-by-step) before starting Jupyter. Set CLINICALKG_PROJECT_ROOT to the parent of the four cohort data folders. A local clinicalKG/run/NACC/ takes precedence over <DATA_ROOT>/NACC/.

## Inputs

Place NACC诊断.csv in <DATA_ROOT>/NACC/ for patient mapping. Put the following classification tables and MRI directories alongside it.

| Clinical CSV | NIfTI directory |
| --- | --- |
| NACC_ad.csv | NACC_nii_ad/ |
| NACC_mci.csv | NACC_mci/ |
| NACC_normal.csv | NACC_nii_no/ |

## Run the notebooks

1. Set CLINICALKG_RAW_KG to the original PrimeKG kg.csv. Open [NACC disease search](NACC_KAMT_DemRKG_disease_search.ipynb) and execute its cells in order to identify candidate entities.
2. Open [NACC seed mapping](NACC_KAMT_DemRKG_seed_mapping.ipynb) to align the clinical fields and prepare NACC_Seed_DementiaHKG.csv in this cohort code directory. Use the [unmatched seed search](NACC_KAMT_DemRKG_unmatched_seed_search.ipynb) to review unresolved terms.
3. Run [NACC graph construction](NACC_KAMT_DemRKG_build.ipynb) to build the graph, create the patient mapping, and train DistMult. The classifier reads NACC-DementiaHKG-Embeddings.npy, NACC-DementiaHKG-Entity2ID.json, and NACC-DementiaHKG-Entities.json from this code folder or <DATA_ROOT>/NACC/DementiaHKG/.
4. Open the [classification notebook](NACC三模态%2BDemRKG.ipynb) in a fresh kernel and execute cells from top to bottom. Its model definitions are in [NACC_model.py](NACC_model.py). Check the resolved paths and match the MRI identifiers to the NIfTI filenames before running training.

## Training configuration

Set CLINICALKG_SEED_COUNT to 5 or 10 before opening the kernel, or set KG_ABLATION_SEEDS to a comma-separated list for a custom run. Restart the kernel when changing environment variables. The diagnostic training section uses 200 epochs and a learning rate of 1e-5. The diagnostic DataLoaders use a batch size of 32. The device selection prefers CUDA when available and falls back to CPU. Generated results are written under clinicalKG/analysis/.
