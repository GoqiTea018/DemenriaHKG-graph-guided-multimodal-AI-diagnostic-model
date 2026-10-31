# clinicalKG：PrimeKGL 与 KAMT_DemRKG

本目录提供基础版 PrimeKGL 和进阶版 KAMT_DemRKG，以及 CustomKG / Full PrimeKG / DemRKG 三种知识来源的统一训练与统计入口。支持 ADNI、AIBL、NACC、PPMI。

## 目录

```text
clinicalKG/
├── PrimeKGL/                  # 基础版：训练 notebook、患者映射、完整图谱预训练
│   ├── README.md
│   ├── Full_PrimeKG.ipynb
│   ├── full_primekg.py
│   ├── ADNI三模态+PrimeKG.ipynb
│   ├── AIBL三模态+PrimeKG.ipynb
│   ├── NACC三模态+PrimeKG.ipynb
│   ├── PPMI三模态+PrimeKG.ipynb
│   ├── Step3-Build-*-PrimeKG.ipynb
│   └── *_model.py
├── KAMT_DemRKG/                      # 进阶版：DemRKG 构建、映射、训练
│   ├── README.md
│   ├── ADNI三模态+DemRKG.ipynb
│   ├── AIBL三模态+DemRKG.ipynb
│   ├── NACC三模态+DemRKG.ipynb
│   ├── PPMI三模态+DemRKG.ipynb
│   ├── *_KAMT_DemRKG_build.ipynb
│   ├── *_KAMT_DemRKG_disease_search.ipynb
│   ├── *_KAMT_DemRKG_seed_mapping.ipynb
│   └── *_model.py
├── run/
│   ├── Run_Experiments.ipynb   # 统一调用三种 KG 的训练、统计和曲线
│   ├── *三模态+CustomKG.ipynb  # 四个数据集的比较基线
│   ├── common/                # 路径、划分、训练、预测和统计支持代码
│   └── record/                # 本地工作记录，文件以数据集名称开头
├── analysis/                  # 本地结果平铺，文件以数据集名称开头
├── shared/FullPrimeKG/        # 本地共享图谱资产
├── tests/
└── requirements.txt
```

方法目录平铺各数据集的 notebook 和模型文件。训练入口继续使用 notebook；`*_model.py` 和 `run/common/` 提供其调用的实现。

## 安装

使用 Python 3.10 或以上版本，安装适合服务器 CUDA 的 PyTorch / torchvision，再从仓库根目录执行：

```bash
python -m pip install -r clinicalKG/requirements.txt
```

用 Jupyter 打开对应 notebook，按顺序执行。统一实验入口为 [run/Run_Experiments.ipynb](run/Run_Experiments.ipynb)。

## 数据与图谱路径

路径集中定义于 [run/common/paths.py](run/common/paths.py)。各数据集先检查 `clinicalKG/run/ADNI/` 等目录；该目录不存在时，读取本机主项目的绝对路径：

```text
C:\Users\HP\Design\DemenriaHKG graph-guided multimodal AI diagnostic model\ADNI
C:\Users\HP\Design\DemenriaHKG graph-guided multimodal AI diagnostic model\AIBL
C:\Users\HP\Design\DemenriaHKG graph-guided multimodal AI diagnostic model\NACC
C:\Users\HP\Design\DemenriaHKG graph-guided multimodal AI diagnostic model\PPMI
```

NACC 使用相同路径规则，补齐数据后即可运行。每个数据集目录包含其临床 CSV、MRI 文件夹和 `CustomKG/`、`PrimeKG/`、`DementiaHKG/` 本地图谱资产。具体 CSV/MRI 名称见 `run/common/config.py`。

迁移到其他机器时，可把数据放在 `run/<数据集>/`，或在启动 Jupyter 前设置 `CLINICALKG_PROJECT_ROOT` 为四个数据集目录的共同父目录。`CLINICALKG_RAW_KG` 指定完整 `kg.csv`；`CLINICALKG_FULL_PRIMEKG_DIR` 指定已有的完整图谱嵌入目录。

Full PrimeKG 嵌入优先读取 `shared/FullPrimeKG/`；已有的 `PrimeKGL/PrimeKG-Embeddings.npy` 等文件也可直接使用。原始图谱和中间产物保留在本地，Git 忽略规则覆盖这些文件。

## 运行

1. **完整 PrimeKG 预训练**：打开 `PrimeKGL/Full_PrimeKG.ipynb`。四个队列共用一次 DistMult 训练结果；中断后设置 `RESUME=True` 恢复。
2. **患者映射与 DemRKG 构建**：按两套方法 README 运行对应建图和映射 notebook。
3. **独立诊断训练**：打开 `PrimeKGL/数据集三模态+PrimeKG.ipynb` 或 `KAMT_DemRKG/数据集三模态+DemRKG.ipynb`。
4. **统一训练与统计**：打开 `run/Run_Experiments.ipynb`，选择数据集和运行模式。

主 notebook 的参数：

| 参数 | 含义 |
|---|---|
| `DATASET_CHOICE` | `1=ADNI`、`2=NACC`、`3=AIBL`、`4=PPMI` |
| `RUN_MODE="check"` | 检查临床数据、MRI 和图谱资产 |
| `RUN_MODE="all"` | 依次训练三种 KG 配置，再汇总统计 |
| `RUN_MODE="train"` | 训练三种配置并汇总指标 |
| `RUN_MODE="stats"` | 默认模式，读取已保存结果并计算统计 |
| `RETRAIN=False` | 复用已完成且配置匹配的训练 |
| `RETRAIN=True` | 重训当前数据集、当前配置并覆盖对应输出 |

三种 KG 分别有训练 cell，可以独立执行。训练使用现有五个种子 `11、23、37、51、67`，沿用原记录级分层划分及最后一轮模型的验证集评估。统计包括 Macro-Precision、Macro-Recall、Macro-F1、Accuracy、AUC，患者级配对 bootstrap 置信区间、置换检验 p 值和 Holm 校正。曲线与预测、权重独立保存。

## 本地输出

`analysis/` 内不设数据集子目录。统一命名为 `数据集_KG名称_文件名`；跨方法统计使用 `数据集_文件名`：

```text
analysis/ADNI_PrimeKG_seed_11.csv
analysis/ADNI_PrimeKG_history_seed_11.csv
analysis/ADNI_PrimeKG_last_seed_11.pt
analysis/ADNI_DementiaHKG_summary.csv
analysis/ADNI_metrics_mean_sd.csv
analysis/ADNI_paired_statistics.csv
analysis/ADNI_training_curves.png
```

历史额外种子、旧协议及归档实验的文件名保留原实验目录标识，分别保存。中间工作记录放在 `run/record/`，以 `ADNI_`、`AIBL_`、`NACC_`、`PPMI_` 开头；共享记录使用 `ALL_`。两处均仅保留在本地。旧到新文件路径映射保存在本地 `run/record/ALL_目录迁移映射_2026-10-31.json`。

## 检查

从仓库根目录执行：

```bash
python -m unittest discover -s clinicalKG/tests -p "test_*.py"
```

检查覆盖 notebook 导入与路径、合成预测的配对统计、训练产物和 Full PrimeKG 断点恢复。
