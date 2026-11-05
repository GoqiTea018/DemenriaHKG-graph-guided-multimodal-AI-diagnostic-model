# PrimeKGL

In VS Code or Jupyter, open [Run_Experiments.ipynb](../run/Run_Experiments.ipynb), select the cohort, and run the notebook cells in order.

See the [run guide](../run/README.md) for cohort files, graph asset paths, and training entry points.

PrimeKGL combines representations learned from the **complete raw PrimeKG** with patient-specific entity mappings for multimodal diagnosis. ADNI, AIBL, NACC, and PPMI share the pretrained graph embeddings and entity index, while each cohort prepares its own patient mappings and trains its diagnostic model.

## Workflow

![PrimeKGL workflow: shared Full PrimeKG embeddings and knowledge-guided multimodal diagnosis](../docs/images/primekgl-workflow.svg)

1. **Encode the complete graph.** Read every triple in the raw `kg.csv`, index its named entities and relations, and train DistMult with a margin-ranking objective and negative tail sampling.
2. **Share graph representations.** Save the entity embedding matrix, entity and relation indices, and resumable training state for reuse across the four cohorts.
3. **Map patients to entities.** Use cohort-specific clinical feature mappings to build each patient's entity sequence and retrieve the corresponding embeddings.
4. **Train the diagnostic model.** Encode EHR, MRI, and biomarkers, incorporate patient-specific knowledge through modality-aware retrieval, and fuse the representations for diagnosis.
5. **Evaluate repeated runs.** Use the shared experiment runner to collect final-epoch validation predictions, metrics, training curves, and paired statistics.

## Files

| File or group | Purpose |
| --- | --- |
| [Full_PrimeKG.ipynb](Full_PrimeKG.ipynb) | Shared graph pretraining with interruption recovery |
| [full_primekg.py](full_primekg.py) | Pretraining implementation used by the notebook |
| `Step3-Build-<DATASET>-PrimeKG.ipynb` | Cohort-specific patient mapping and shared asset loading |
| Cohort diagnostic training notebooks | Standalone training for ADNI, AIBL, NACC, and PPMI |
| `<DATASET>_model.py` | Cohort-specific model implementations |

## Dataset

The [shared dataset guide](../README.md#dataset-organization) describes the ADNI, AIBL, NACC, and PPMI files and the clinical CSV / MRI directory layout. PrimeKGL uses these cohort inputs together with the complete raw PrimeKG and shared graph embeddings.

## Usage

1. Install dependencies and configure cohort data and raw `kg.csv` paths using the [clinicalKG guide](../README.md).
2. Run [Full_PrimeKG.ipynb](Full_PrimeKG.ipynb) to generate shared graph representations, or configure an existing Full PrimeKG embedding directory.
3. Run the patient-mapping cells in the appropriate notebook: [ADNI](Step3-Build-ADNI-PrimeKG.ipynb), [AIBL](Step3-Build-AIBL-PrimeKG.ipynb), [NACC](Step3-Build-NACC-PrimeKG.ipynb), or [PPMI](Step3-Build-PPMI-PrimeKG.ipynb).
4. Open the cohort's diagnostic training notebook in this directory, or use [Run_Experiments.ipynb](../run/Run_Experiments.ipynb) for the shared comparison.

Set `RESUME=True` in [Full_PrimeKG.ipynb](Full_PrimeKG.ipynb) to continue pretraining.

## Pretraining settings and outputs

| Setting | Default |
| --- | --- |
| Embedding model | DistMult |
| Embedding dimension | 128 |
| Batch size | 512 |
| Learning rate | 0.005 |
| Pretraining epochs | 50 |

Pretraining uses the complete input graph's edge records and named entity and relation vocabulary. Shared outputs are stored locally in `clinicalKG/shared/FullPrimeKG/`:

- `PrimeKG-Embeddings.npy`: entity representations.
- `PrimeKG-Entity2ID.json` and `PrimeKG-Relation2ID.json`: graph indices.
- `PrimeKG-DistMult-last.pt` and `PrimeKG-TrainingState.json`: checkpoint and training metadata.
- `PrimeKG-LossHistory.json` and `PrimeKG-LossCurve.png`: pretraining loss records.

Existing `PrimeKG-*` assets in this method directory are also supported by the shared path resolver. Diagnostic predictions, checkpoints, metrics, and curves are stored in the local `clinicalKG/analysis/` directory. See the [evaluation and output guide](../README.md#evaluation-and-statistics) for the shared protocol.
