# DemRKGL

In VS Code or Jupyter, open [Run_DemRKGL.ipynb](clinicalKG/run/Run_DemRKGL.ipynb), select the cohort, and run the notebook cells in order.

- [Step-by-step cohort guides](clinicalKG/KAMT_DemRKG/README.md#run-one-cohort-step-by-step) cover graph preparation and individual notebook execution for ADNI, AIBL, NACC, and PPMI.
- [Batch training guide](clinicalKG/run/README.md) explains how Run_DemRKGL.ipynb checks the selected cohort inputs and launches diagnostic training across the chosen seed preset using prepared graph assets.
- [Cohort data and local paths](clinicalKG/README.md) lists the clinical tables and MRI directories.

## Compare DemRKG and Full PrimeKG

Use the [clinicalKG comparison guide](https://github.com/GoqiTea018/DemRKG-model/blob/clinicalKG/clinicalKG/README.md) to run both methods under the shared protocol. The comparison reports accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest ROC AUC on the epoch-200 validation split. The paired analysis reports metric differences on aligned predictions, bootstrap confidence intervals, permutation p-values, and Holm-adjusted p-values.
