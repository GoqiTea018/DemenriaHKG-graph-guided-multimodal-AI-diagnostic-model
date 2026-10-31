# PrimeKGL：基础版

PrimeKGL 在完整原始 PrimeKG 上训练一套共享 DistMult 嵌入，结合各数据集的患者实体映射进行三模态诊断。四个数据集共用图谱嵌入及实体索引，分别生成患者查询和训练诊断模型。

## 文件

- `Full_PrimeKG.ipynb`：共享图谱预训练 notebook，包含中断保存和恢复功能。
- `full_primekg.py`：相同预训练器的 Python 实现与命令行入口。
- `Step3-Build-数据集-PrimeKG.ipynb`：各数据集患者映射及共享资产调用。
- `数据集三模态+PrimeKG.ipynb`：ADNI、AIBL、NACC、PPMI 诊断训练入口。
- `数据集_model.py`：保留各队列现有模型实现。

## 使用

1. 按 [总说明](../README.md) 配置临床数据和完整 `kg.csv`。
2. 打开 `Full_PrimeKG.ipynb` 训练共享嵌入；已有完整图谱嵌入时直接使用本地文件。
3. 打开相应的 `Step3-Build-数据集-PrimeKG.ipynb`，执行患者映射相关 cell。
4. 打开 `数据集三模态+PrimeKG.ipynb` 独立训练，或通过 [统一实验入口](../run/Run_Experiments.ipynb) 调用。

从仓库根目录运行完整图谱预训练器：

```bash
python -m clinicalKG.PrimeKGL.full_primekg --raw-kg kg.csv --epochs 50 --device cuda
python -m clinicalKG.PrimeKGL.full_primekg --raw-kg kg.csv --epochs 50 --device cuda --resume
```

默认 DistMult 为 128 维、batch size 512、学习率 0.005、50 epochs。训练使用完整输入图谱的实体、关系和边记录。共享输出位于本地 `clinicalKG/shared/FullPrimeKG/`，已有的本目录 `PrimeKG-*` 中间文件继续保留并可读取。诊断训练结果平铺保存到本地 `clinicalKG/analysis/`。
