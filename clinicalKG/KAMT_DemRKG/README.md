# DemRKGL (KAMT_DemRKG)

Use the cohort folders below for step-by-step execution. The [batch guide](../run/README.md) covers training with prepared graph assets.

DemRKGL combines a clinically aligned **DemRKG** with knowledge-guided multimodal diagnosis. This directory, named `KAMT_DemRKG` in the codebase, contains disease search, clinical seed alignment, graph construction, patient mapping, DistMult pretraining, and diagnostic training for ADNI, AIBL, NACC, and PPMI.

## Workflow

![DemRKG construction: clinical seed alignment, degree-aware pruning, and entity representations](../docs/images/demrkg-workflow.png)

The figure summarizes three stages that produce the graph representations used by the diagnostic model:

1. **Clinical seed alignment.** Extract clinical keywords from cohort-specific diagnostic records and assessment scales. Align these fields with PrimeKG entities and combine them with core seeds identified through medical domain knowledge.
2. **Degree-aware graph pruning.** Retrieve one-hop neighborhoods, apply entity-type and relation filters, expand to two-hop candidates, and perform source-wise top-k selection. Candidate ranking combines local connectivity with a penalty based on global graph degree to obtain the refined triples.
3. **Relation-trained entity representations.** Train DistMult on the refined triples with a margin-ranking objective. Export the entity embedding matrix and entity index for patient-specific knowledge activation.

The diagnostic stage retrieves each patient's mapped entity embeddings, integrates them with EHR, MRI, and biomarker representations, and fuses the resulting features for classification.

## Files

| File or group | Purpose |
| --- | --- |
| `<DATASET>_KAMT_DemRKG_disease_search.ipynb` | Search for relevant disease entities |
| `<DATASET>_KAMT_DemRKG_seed_mapping.ipynb` | Align clinical fields with seed entities |
| `<DATASET>_KAMT_DemRKG_build.ipynb` | Construct DemRKG, map patients, and train DistMult |
| Cohort diagnostic training notebooks | Standalone training for ADNI, AIBL, NACC, and PPMI |
| `<DATASET>_model.py` | Cohort-specific model implementations |

## Dataset

The [shared dataset guide](../README.md#dataset-organization) describes the ADNI, AIBL, NACC, and PPMI files and the clinical CSV / MRI directory layout. DemRKGL uses the same cohort inputs as PrimeKGL, together with clinical seed mappings and the constructed DemRKG representations.

## Run one cohort step by step

From the repository root, install the dependencies and start Jupyter with your data paths.

~~~powershell
python -m pip install -r clinicalKG/requirements.txt
$env:CLINICALKG_PROJECT_ROOT = "D:/data/cohorts"
$env:CLINICALKG_RAW_KG = "D:/data/PrimeKG/kg.csv"
$env:CLINICALKG_SEED_COUNT = "10"
python -m jupyterlab
~~~

Open the guide for your cohort and execute its notebooks in order.

- [ADNI](ADNI/README.md)
- [AIBL](AIBL/README.md)
- [NACC](NACC/README.md)
- [PPMI](PPMI/README.md)

For batch diagnostic training, open [Run_DemRKGL.ipynb](../run/Run_DemRKGL.ipynb), choose DATASET and SEED_COUNT, and run the configuration cell with TRAIN = False to check inputs. Then set TRAIN = True and run the cell again.


## Graph settings and local assets

Construction notebooks define the entity-type and relation filters, two-hop scoring, top-k selection, and embedding settings. The source-wise selection uses **k = 5 for ADNI and NACC** and **k = 6 for AIBL and PPMI**.

Local assets include the clinical seed tables (`*_Seed_DementiaHKG.csv`), graph CSV files, patient-to-entity mappings, embeddings, and entity indices. The training workflow uses the name `DemRKG`; existing graph assets and the shared experiment runner use the internal identifier `DementiaHKG`.

Diagnostic predictions, checkpoints, metrics, and curves are saved in `clinicalKG/analysis/`. See the [shared evaluation guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/run/README.md#recorded-results) for final-epoch validation metrics and paired statistical analysis.

## Compare DemRKG and Full PrimeKG

Use the [clinicalKG comparison guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/README.md) to run both methods under the shared protocol. The comparison reports accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest ROC AUC on the epoch-200 validation split. The paired analysis reports metric differences on aligned predictions, bootstrap confidence intervals, permutation p-values, and Holm-adjusted p-values.
