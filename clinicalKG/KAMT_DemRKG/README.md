# KAMT_DemRKG：进阶版

KAMT_DemRKG 将面向临床特征构建的 DemRKG 与三模态诊断模型结合。本目录由原 `DemRKL/` 中的现有代码整理而来，包含疾病检索、seed 对齐、子图构建、患者映射、图谱嵌入和诊断训练。

## 文件

- `数据集三模态+DemRKG.ipynb`：ADNI、AIBL、NACC、PPMI 的现有训练 notebook。
- `数据集_KAMT_DemRKG_disease_search.ipynb`：疾病实体检索。
- `数据集_KAMT_DemRKG_seed_mapping.ipynb`：临床字段与 seed 对齐。
- `NACC_KAMT_DemRKG_unmatched_seed_search.ipynb`：NACC 未匹配 seed 检索。
- `数据集_KAMT_DemRKG_build.ipynb`：DemRKG 构建、患者映射和 DistMult 训练。
- `数据集_model.py`：各队列现有模型实现。

## 使用

1. 按 [总说明](../README.md) 配置数据目录、原始 PrimeKG 和本地 seed 映射文件。
2. 已有 seed 映射时，直接使用 `数据集_KAMT_DemRKG_build.ipynb`。重新生成映射时，依次运行疾病检索、seed 对齐及建图 notebook。
3. 打开 `数据集三模态+DemRKG.ipynb` 独立训练，或通过 [统一实验入口](../run/Run_Experiments.ipynb) 与 CustomKG、Full PrimeKG 一起训练和比较。

建图沿用现有筛选、两跳评分、Top-k 和嵌入设置；ADNI/NACC 的 k 为 5，AIBL/PPMI 为 6。训练 notebook 使用 `DemRKG` 命名；现有图谱资产仍通过内部 `DementiaHKG` 标识读取。

`*_Seed_DementiaHKG.csv`、图谱 CSV、患者实体 JSON、嵌入及索引保存在本地。本目录优先使用已有资产，也可从数据目录下的 `DementiaHKG/` 读取。诊断训练、预测与统计结果平铺保存到本地 `clinicalKG/analysis/`。
