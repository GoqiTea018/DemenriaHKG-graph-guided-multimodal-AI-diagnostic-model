# clinicalKG: Knowledge-Guided Multimodal Diagnosis

Knowledge-guided diagnostic models that combine electronic health records (EHR), magnetic resonance imaging (MRI), and biomarkers (BM) across **ADNI, AIBL, NACC, and PPMI**.

This branch provides two knowledge pipelines and a shared experiment runner for comparison with CustomKG.

| Project | Knowledge source | Documentation |
| --- | --- | --- |
| **PrimeKGL** | Shared DistMult representations learned from the complete raw PrimeKG | [Workflow and usage](clinicalKG/PrimeKGL/README.md) |
| **DemRKGL (KAMT_DemRKG)** | Clinically aligned DemRKG subgraphs with degree-aware pruning | [Workflow and usage](clinicalKG/KAMT_DemRKG/README.md) |

## PrimeKGL workflow

![PrimeKGL workflow: complete PrimeKG pretraining, patient mapping, and multimodal diagnosis](clinicalKG/docs/images/primekgl-workflow.svg)

The complete PrimeKG is encoded once with DistMult. Each cohort maps patient clinical features to graph entities, retrieves their embeddings, and combines the resulting knowledge sequences with EHR, MRI, and biomarker representations for diagnosis.

## DemRKGL workflow

![DemRKG construction: clinical seed alignment, degree-aware pruning, and entity representations](clinicalKG/docs/images/demrkg-workflow.png)

Clinical fields and medical domain knowledge identify seed entities in PrimeKG. Neighborhood retrieval, type and relation filtering, and source-wise top-k selection produce DemRKG. DistMult embeddings support patient-specific knowledge activation and the downstream multimodal diagnostic model.

## Dataset

The project's MRI dataset is available on [Zenodo (record 19646995)](https://zenodo.org/records/19646995), DOI: [10.5281/zenodo.19646995](https://doi.org/10.5281/zenodo.19646995). The release provides ADNI, AIBL, and PPMI archives and three class-specific NACC archives.

The [dataset guide](clinicalKG/README.md#dataset-download-and-organization) lists download links, archive sizes, clinical CSV / MRI folder pairs, and extraction instructions. Both PrimeKGL and DemRKGL use the same cohort data.

## Getting started

1. Install dependencies and configure data paths using the [clinicalKG setup guide](clinicalKG/README.md).
2. Prepare graph embeddings and patient mappings using the method-specific workflows above.
3. Open [Run_Experiments.ipynb](clinicalKG/run/Run_Experiments.ipynb) to check inputs, train CustomKG / Full PrimeKG / DemRKG configurations, and summarize metrics and paired statistics.

The [clinicalKG guide](clinicalKG/README.md) describes the directory layout, runtime options, evaluation protocol, and output files.
