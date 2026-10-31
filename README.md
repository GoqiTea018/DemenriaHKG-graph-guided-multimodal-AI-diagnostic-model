# PrimeKGL / KAMT_DemRKG

面向 ADNI、AIBL、NACC、PPMI 的知识图谱引导三模态诊断代码。

- **[PrimeKGL](clinicalKG/PrimeKGL/README.md)**：基础版，使用完整 PrimeKG 的共享 DistMult 嵌入与患者实体映射。
- **[KAMT_DemRKG](clinicalKG/KAMT_DemRKG/README.md)**：进阶版，包含 DemRKG 构建、患者映射与知识引导多模态训练。
- **[运行说明](clinicalKG/README.md)**：安装、数据路径、独立训练及三种 KG 的统一比较。
- **[实验主入口](clinicalKG/run/Run_Experiments.ipynb)**：调用 CustomKG、PrimeKG、DementiaHKG 训练 notebook，汇总五项指标、置信区间和 p 值。

各方法训练入口保持 `数据集三模态+KG名称.ipynb` 命名。此分支发布 README、依赖配置、Python 支持代码和 notebook；数据、图谱中间文件、权重、实验结果与工作记录保存在本地。
