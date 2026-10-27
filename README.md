# KAMT: Knowledge-Augmented Multimodal Synergistic Interaction

KAMT 将临床记录（EHR）、生物标志物（BM）、MRI 与知识图谱结合，用于 ADNI、AIBL、NACC、PPMI 四个队列的诊断实验。本仓库包含知识图谱构建、DistMult 嵌入学习、多模态分类、五次独立训练与配对统计，以及 DemRKG 语义子图绘制代码。

## 目录与入口

| 目录 | 内容与使用说明 |
|---|---|
| [clinicalKG/](clinicalKG/README.md) | 当前 Full PrimeKG、DemRKG、CustomKG 比较流程，包含独立脚本、模型副本、notebook、测试与依赖列表 |
| [experiments/](experiments/README.md) | 原队列图谱实验的统一检查、训练和统计入口，以及共享患者划分工具 |
| [semantic_subgraph/](semantic_subgraph/README.md) | 从各队列 DemRKG CSV 生成带实体名称及关系标签的论文图 |
| ADNI/、AIBL/、NACC/、PPMI/ | 各队列原始建图与分类 notebook、model1.py、CustomKG/DementiaHKG/PrimeKG 目录 |
| SOTA/ | 对比模型相关代码 |

代码中的 `DementiaHKG` 对应论文中的 DemRKG。`clinicalKG/PrimeKGL` 使用完整原始 PrimeKG；`clinicalKG/DemRKL` 保留疾病相关子图构建流程。原 `experiments/` 读取各队列已有 PrimeKG 资产，两种实验入口分别组织结果。

## 图谱与规模

[PrimeKGL README](clinicalKG/PrimeKGL/README.md) 与 [DemRKL README](clinicalKG/DemRKL/README.md) 分别提供运行流程、图谱可视化和运行方式。[图谱规模审计](clinicalKG/GRAPH_STATISTICS.md) 提供完整规模、实体类型和关系类型统计。

| 图谱 | 实体（名称索引） | 实体类型 | 边记录 | 去重有向三元组 | 关系类型 |
|---|---:|---:|---:|---:|---:|
| FullPrimeKG | 129,262 | 10 | 8,100,498 | 8,099,991 | 30 |
| ADNI | 4,099 | 8 | 23,630 | 23,630 | 17 |
| AIBL | 3,894 | 8 | 22,621 | 22,621 | 18 |
| NACC | 6,547 | 8 | 42,909 | 42,909 | 17 |
| PPMI | 2,497 | 8 | 13,050 | 13,050 | 17 |

## 环境与数据

使用 Python 3.10 或以上版本，先安装适合服务器 CUDA 的 PyTorch/torchvision，再安装：

```bash
pip install -r clinicalKG/requirements.txt
```

在仓库根目录准备完整原始 `kg.csv`；各队列的临床 CSV 和 MRI 放入相应数据集目录。具体文件名与图谱资产见两个实验目录的 README。患者数据、原始图谱、模型权重、嵌入和逐患者预测保存在本地，由 `.gitignore` 排除。仓库保留建图所需的字段—实体种子字典，以及 `clinicalKG/DemRKL/` 下四个队列的公开 DemRKG 图谱 CSV。

## 推荐流程：Full PrimeKG 比较

1. 打开 [clinicalKG/Full_PrimeKG.ipynb](clinicalKG/Full_PrimeKG.ipynb)，或从仓库根目录执行一次完整图谱预训练：

   ```bash
   python clinicalKG/full_primekg.py --raw-kg kg.csv --epochs 50 --device cuda
   # 中断后继续
   python clinicalKG/full_primekg.py --raw-kg kg.csv --epochs 50 --device cuda --resume
   ```

   四个队列共用 `clinicalKG/shared/FullPrimeKG/` 中的 DistMult 实体嵌入和索引。

2. 按 [clinicalKG 说明](clinicalKG/README.md) 为所选队列生成患者实体映射，准备 CustomKG 与 DemRKG 资产。
3. 打开 [clinicalKG/Run_Experiments.ipynb](clinicalKG/Run_Experiments.ipynb)，设置 `DATASET_CHOICE`：`1=ADNI`、`2=NACC`、`3=AIBL`、`4=PPMI`。运行配置及输入检查，再分别执行 CustomKG、Full PrimeKG、DemRKG 的训练单元格，最后运行统计及曲线单元格。

主比较结果保存到 `clinicalKG/results_full_primekg/数据集/`。单独的 `*_train.py` 入口默认写入 `clinicalKG/results/`；原队列实验入口写入 `experiments/results/`。重新运行主 notebook 的训练单元格会覆盖对应配置的已有结果。

2026-10-27 的 clinicalKG 更新包含 AIBL/PPMI 的实体填充掩码修正、全填充序列回退与 PPMI MRI 文件名检查修正。AIBL 三种配置应使用更新后的模型重新训练；已有核验记录见 [clinicalKG/record/](clinicalKG/record/)。

## 原队列实验入口

从仓库根目录执行，按需选择一个队列：

```bash
python -m experiments.run --dataset ADNI --stage check
python -m experiments.run --dataset ADNI --stage all
python -m experiments.run --dataset ADNI --stage stats
```

`check` 检查输入，`train` 训练，`stats` 读取已保存预测进行统计，`all` 依次训练和统计。完整参数与资产说明见 [experiments/README.md](experiments/README.md)。

## 实验划分与统计

分类 notebook 采用患者级分层固定划分，约为 train/validation/test = 64%/16%/20%，同一患者的多次记录归入同一集合。五次独立训练使用种子 `11、23、37、51、67`，共用固定划分，并按验证损失选择 checkpoint。结果报告均值与样本标准差；统计工具提供患者级配对差值、95% CI、置换 p 值和 Holm 校正。

PPMI 沿用前 120 条 PD 记录的输入设置；AIBL 无 KG 基线采用完整队列。依赖训练集常模的患者激活应与当前划分配套生成。原队列 notebook 的 CustomKG 使用 32 维嵌入，另外两种图谱使用 128 维，比较时需结合模型容量设置解释结果。

## 验证与绘图

```bash
python -m unittest discover -s clinicalKG/tests -v
python -m pytest experiments/tests -q -p no:cacheprovider
python semantic_subgraph/plot_semantic_subgraph.py --dataset NACC
```

测试覆盖合成图预训练与续训、路径解析、患者划分和配对统计。真实训练由使用者在数据与计算环境就绪后启动；论文结果以对应运行保存的预测和运行记录为依据。
