# 原队列图谱实验脚本运行说明

此入口使用各队列原有的 CustomKG、PrimeKG 和 DementiaHKG 资产。完整原始 PrimeKG 的共享预训练与三种 KG 比较使用 [clinicalKG/Run_Experiments.ipynb](../clinicalKG/Run_Experiments.ipynb)，详见 [clinicalKG README](../clinicalKG/README.md)。

每次由你选择一个数据集：`ADNI`、`AIBL`、`NACC` 或 `PPMI`。命令只处理所选数据集。

## 1. 文件用途

| 文件 | 用途 | 单独运行会训练吗？ |
|---|---|---|
| `run.py` | 统一入口：检查、训练、统计 | 由 `--stage` 决定 |
| `worker.py` | 在独立进程中执行选定队列的 notebook，到五次训练结束 | 由 `run.py` 调用 |
| `config.py` | 统一维护数据集、MRI 路径、KG 文件名和五个种子 | 否 |
| `experiment_split.py` | 为各模型和临床常模提供同一患者划分 | 否 |
| `kg_ablation.py` | 五次训练、验证集选模型、导出测试预测 | notebook 调用时训练 |
| `kg_ablation_stats.py` | 根据已有预测计算配对 CI、p 值和 Holm 校正 | 否 |
| `tests/` | 小型验证测试 | 使用测试数据，不运行论文完整模型 |

## 2. 从哪里启动

在 PowerShell/终端中先进入项目根目录，并使用已安装 PyTorch 等依赖的 Python 环境：

```powershell
cd "你的仓库目录"
```

所有命令使用 `python -m experiments.run`，无需把脚本移回主目录。

## 3. 自己选择数据集

先运行相应的检查命令：

```powershell
# 选 ADNI
python -m experiments.run --dataset ADNI --stage check

# 选 AIBL
python -m experiments.run --dataset AIBL --stage check

# 选 NACC
python -m experiments.run --dataset NACC --stage check

# 选 PPMI
python -m experiments.run --dataset PPMI --stage check
```

检查仅核对依赖、CSV、MRI 文件路径、KG 资产和训练入口，不读取 MRI 体数据，不开始训练。

数据就绪后，运行所选数据集的一条命令：

```powershell
# 四条命令按需要选一条
python -m experiments.run --dataset ADNI --stage all
python -m experiments.run --dataset AIBL --stage all
python -m experiments.run --dataset NACC --stage all
python -m experiments.run --dataset PPMI --stage all
```

这里的 `all` 指**所选数据集的训练＋统计流程**。每条命令依次运行 CustomKG、PrimeKG、DementiaHKG 三种配置，每种训练五次，即总计 15 次分类训练。`DementiaHKG` 是代码中 DemRKG 的名称。

五个训练种子为 `11、23、37、51、67`。划分固定，按验证损失选择每次运行的模型，然后导出测试集结果。epoch、学习率、模型宽度等设置沿用相应 notebook，当前训练上限为 200 epochs。

## 4. 每个数据集需要什么

| 数据集 | 队列 CSV | MRI 目录（位于对应数据集目录内） |
|---|---|---|
| ADNI | `AD.csv`、`mci.csv`、`normal.csv` | `ad_nii_KG`、`NC_nii_KG`、`normal_nii_kg` |
| AIBL | `AD.csv`、`MCI.csv`、`NC.csv` | `AD`、`MCI`、`NC` |
| NACC | `NACC_ad.csv`、`NACC_mci.csv`、`NACC_normal.csv` | `NACC_nii_ad`、`NACC_mci`、`NACC_nii_no` |
| PPMI | `PD1.csv`、`control.csv`、`prodromal.csv`、`swedd.csv` | `PD`、`Control`、`Prodromal`、`SWEDD` |

NACC 的 MRI 较大，当前本机未放入数据；把文件放回原代码对应目录即可，不需要改变路径。

还需要三种 KG 已生成的嵌入和映射文件。入口优先读取各数据集 `CustomKG/`、`PrimeKG/`、`DementiaHKG/` 内的资产，其次读取数据集目录下的同名文件；无需手动复制新生成文件到主目录。检查命令会给出缺少文件的具体位置。

- CustomKG：`CustomKG-数据集.npy` 与 `CustomKG-数据集.txt`。
- DemRKG：`数据集-DementiaHKG-Embeddings.npy`、`Entity2ID.json` 与 `Entities.json`。
- PrimeKG：NACC/PPMI 使用 `数据集-PrimeKG-…`；ADNI/AIBL 使用 `数据集-PrimeKG-Raw-…`。

训练入口消费已有图谱资产。图谱构建与嵌入学习仍使用原建图 notebook。采用新的固定划分重跑前，应更新依赖训练常模的患者激活文件；NACC 的建图 notebook 已接入同一划分。

NACC 的 CustomKG 和 DemRKG notebook 已接入五次训练导出；PrimeKG 在统一入口中复用 DemRKG 的架构模板，自动切换到 PrimeKG 嵌入、实体映射和患者序列，并以 `PrimeKG` 保存结果。

## 5. 只训练或只统计

```powershell
# 只训练你选择的数据集
python -m experiments.run --dataset NACC --stage train

# 已有预测文件时，只计算统计结果，不重新训练，也不需要加载 MRI
python -m experiments.run --dataset NACC --stage stats

# 只训练一类 KG，方便分别完成三种配置
python -m experiments.run --dataset NACC --stage train --graphs PrimeKG
```

`--stage stats` 会比较 DemRKG 与 CustomKG、PrimeKG，因此需要三种配置各五份预测文件。默认主要指标为 macro-F1，bootstrap 和置换各进行 10,000 次。可显式选择补充指标：

```powershell
python -m experiments.run --dataset NACC --stage stats --metrics f1 accuracy auc
```

统计采用患者级配对抽样/交换，同一患者的多次记录和五个训练种子的预测一同处理。每个数据集、每个指标的两个 KG 比较应用 Holm 校正。输出反映已训练模型在当前测试患者上的性能差异。

## 6. 结果保存位置

默认保存到 `experiments/results/所选数据集/`：

```text
experiments/results/NACC/
  CustomKG/
    seed_11.csv ... seed_67.csv   每次训练的逐患者预测与概率
    summary.csv                 五次运行的各项指标
  PrimeKG/
    ...
  DementiaHKG/
    ...
  metrics_mean_sd.csv            三种配置的 mean、sample SD
  paired_f1_patient.csv          F1 差值、95% CI、p 值、Holm 校正 p 值
```

选择多个统计指标时，会相应生成 `paired_accuracy_patient.csv`、`paired_auc_patient.csv` 等文件。

可以指定输出目录；后续统计时使用相同目录：

```powershell
python -m experiments.run --dataset NACC --stage all --output "D:\KAMT-results\run01"
python -m experiments.run --dataset NACC --stage stats --output "D:\KAMT-results\run01"
```

**运行 `--stage all` 可以完成当前三种 KG 的训练与统计。** 它不包含 SOTA、KAMP 和新增组件消融；这些实验有各自 notebook，当前共享患者划分。该入口也不会修改论文表格。CustomKG 的既有 32 维设置与另外两种图谱的 128 维设置仍沿用当前 notebook，统一维度后的控制实验需另行配置与重跑。

## 7. 验证与历史检查

从仓库根目录运行 `python -m pytest experiments/tests -q -p no:cacheprovider`，验证患者划分、入口适配与统计工具。依赖安装见 [主 README](../README.md)。


2026-10-25，本地 ADNI/AIBL/PPMI 的 CSV、MRI 路径和三种 KG 资产通过存在性检查；NACC 的 KG 资产就绪，MRI 留待放回原目录。这条记录对应当日的输入存在性检查；后续训练状态以各结果目录中的预测、汇总和运行记录为准。
