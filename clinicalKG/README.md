# clinicalKG：PrimeKGL 与 DemRKL

这里集中维护四个队列的建图、多模态训练与统计代码，包含 Full PrimeKG 的共享预训练入口和各队列独立的患者映射、诊断模型。源码与主目录说明见 [项目 README](../README.md)。

- **PrimeKGL**：完整原始 PrimeKG → 共享 DistMult 实体嵌入 → 各队列患者实体激活 → 原多模态诊断训练。
- **DemRKL**：保留各队列原有的 seed、黑名单、两跳评分、Top-k、DistMult 和多模态训练。
- **Full PrimeKG 只训练一次**。四个队列读取相同的实体嵌入及实体索引；患者 `Entities.json` 和诊断模型分别生成/训练。

## 图谱与规模统计

- [PrimeKGL：Full PrimeKG 流程和图谱](PrimeKGL/README.md)
- [DemRKL：DemRKG 构建和四队列图谱](DemRKL/README.md)
- [完整图谱规模、实体类型及关系数量](GRAPH_STATISTICS.md)

| 图谱 | 实体（名称索引） | 实体类型 | 边记录 | 去重有向三元组 | 关系类型 |
|---|---:|---:|---:|---:|---:|
| FullPrimeKG | 129,262 | 10 | 8,100,498 | 8,099,991 | 30 |
| ADNI | 4,099 | 8 | 23,630 | 23,630 | 17 |
| AIBL | 3,894 | 8 | 22,621 | 22,621 | 18 |
| NACC | 6,547 | 8 | 42,909 | 42,909 | 17 |
| PPMI | 2,497 | 8 | 13,050 | 13,050 | 17 |

以上统计来自实际图谱 CSV。两份方法 README 附有论文现有的 KAMT 框架图、DemRKG 流程图、PNG/SVG 图谱和可追溯的展示子图 JSON；四个 DemRKG 详细图谱 CSV 位于各队列 DemRKL 目录。

## 1. 目录与文件名

**训练与统计主入口：`Run_Experiments.ipynb`。** 顶部 `DATASET_CHOICE` 取 `1` 为 ADNI、`2` 为 NACC、`3` 为 AIBL、`4` 为 PPMI。分别运行 CustomKG、Full PrimeKG、DemRKG 各五次诊断训练，再计算 Macro-Precision、Macro-Recall、Macro-F1、Accuracy、AUC 的患者级配对差值、95% CI、p 值及 Holm 校正。最后一个单元格展示训练集/验证集曲线。Full PrimeKG 图谱嵌入读取已有文件，识别共享目录或 `PrimeKGL/` 根目录。所有队列均保存到 `results_full_primekg/`，沿用原文件名；重新运行训练 cell 会覆盖对应配置的原结果。`RUN_MODE="stats"` 重算当前结果目录中已保存预测的统计结果。

2026-10-27 更新：AIBL/PPMI 的模型直接使用 `False=有效实体、True=填充位` 的掩码，全填充序列使用零向量回退；三种 KG 配置均加载 `clinicalKG` 中的修正模型副本。AIBL 需由用户自行重新训练三种配置，并覆盖原目录中的结果；修正前聚合数值保留在 record 报告中。PPMI 输入检查按原训练脚本读取 CSV 第二列中的 MRI 文件名。主 notebook 的配置 cell 会自动重新加载 clinicalKG 模块，并将工作目录切换到整个主文件夹。更新后重新执行配置 cell，再按顺序运行；完整训练由用户自行启动。若 notebook 从项目目录之外启动，可在配置中指定 MAIN_FOLDER。修复分析见 `record/ADNI_FullPrimeKG_运行结果分析_2026-10-27.md`。

三种 KG 的诊断训练分别放在三个独立 cell：CustomKG、Full PrimeKG、DemRKG。每个 cell 只训练对应配置，并实时打印每轮的种子、epoch、训练/验证 Loss、Accuracy、Precision、Recall、F1、AUC 和最佳 epoch（Precision/Recall/F1 为 macro）。可以分别运行；三种配置完成后再运行统计 cell。最后一个 cell 绘制曲线。

```text
你的主文件夹/
  kg.csv                         # 云服务器上推荐放完整原始 PrimeKG 的位置
  ADNI/                          # 原临床 CSV、MRI
  AIBL/
  NACC/
  PPMI/
  clinicalKG/
    full_primekg.py              # 一次性 Full PrimeKG 训练入口
    paths.py                     # 唯一的主目录/图谱路径配置
    notebook_runner.py
    experiments/                 # 原划分和五次训练工具的独立副本
    shared/FullPrimeKG/          # 四个队列共享的 Full PrimeKG 训练结果
    PrimeKGL/
      ADNI_PrimeKGL/
      AIBL_PrimeKGL/
      NACC_PrimeKGL/
      PPMI_PrimeKGL/
    DemRKL/
      ADNI_DemRKL/
      AIBL_DemRKL/
      NACC_DemRKL/
      PPMI_DemRKL/
    results_full_primekg/        # Run_Experiments.ipynb 三种 KG 的比较结果
    results/                     # 单独 *_train.py 入口的结果
    record/                      # 运行分析及修正核验记录
```

每个 `数据集_方法/` 目录包含：

- `数据集_方法_build.py`：建图/患者映射/KGE 的分阶段命令入口。
- `数据集_方法_train.py`：原多模态分类训练的命令入口。
- 可编辑的建图和分类 notebook、该队列原 `model1.py`、已确定的 seed 映射 CSV。
- DemRKL 另外保留疾病检索、seed 对齐 notebook；NACC 还保留未匹配 seed 检索。

PrimeKGL 中的 **`Step3-Build-数据集-PrimeKG.ipynb` 和 `数据集三模态+PrimeKG.ipynb` 保留原名**。原项目没有独立 NACC PrimeKG 分类 notebook，因此这里按照原 `experiments.worker` 的做法，复制 NACC DementiaHKG 架构模板并切换 PrimeKG 资产，命名为 `NACC三模态+PrimeKG.ipynb`。

DemRKL 的 notebook 统一命名为 `数据集_DemRKL_build.ipynb`、`数据集_DemRKL.ipynb`。其生成文件继续使用原有 `DementiaHKG` 名称，保持原训练流程兼容。

`source_manifest.json` 记录了每份代码的来源和原始文件 SHA-256。`tools/organize_sources.py` 是此次复制整理脚本；日常运行无需执行，重新执行会重新生成这些副本。

## 2. 主目录与相对路径

`paths.py` 默认将 `clinicalKG` 的上一级作为你的主文件夹：

```python
PROJECT_ROOT = CLINICALKG_ROOT.parent
```

临床 CSV/MRI 从 `PROJECT_ROOT / 数据集` 读取。队列图谱输出写到对应的 `数据集_方法/`；Full PrimeKG 的公共资产写到 `shared/FullPrimeKG/`。

完整图谱优先读取主目录 `kg.csv`，也支持 `shared/FullPrimeKG/kg.csv`。本机为方便直接使用，会依次查找原队列 `PrimeKG/kg.csv` 或 `DementiaHKG/kg.csv`；读取的是这些文件中的完整图谱，不使用原先的 `*-Raw-G.csv` 或两跳子图。

需要在其他位置读取数据时，可设置 `CLINICALKG_PROJECT_ROOT`。`CLINICALKG_RAW_KG` 可指定原始图谱；相对值以主文件夹为基准。

整个 `clinicalKG` 可随主文件夹移动到 Linux 云服务器，代码没有写死本机盘符。仅运行 Full PrimeKG 预训练时，只需上传整个 `clinicalKG/` 和完整 `kg.csv`，不需要患者数据或 MRI。

## 3. 云服务器：只训练一次 Full PrimeKG

**Jupyter 用户直接打开 `Full_PrimeKG.ipynb`。** 该 notebook 内置完整训练与中断保存代码，可单独上传 notebook 和完整 `kg.csv`，按单元格顺序执行。首次 `RESUME=False`，中断后改成 `True` 再运行；参数和相对路径都在第一个代码单元格。输出仍为四队列共用的 `shared/FullPrimeKG/`。

下面的命令均从**你的主文件夹**运行。使用 Python 3.10 或以上版本，安装服务器对应的 PyTorch CUDA 版本及 `clinicalKG/requirements.txt` 中依赖。

```bash
python clinicalKG/full_primekg.py --raw-kg kg.csv --epochs 50 --device cuda
```

默认参数：128 维、batch size 512、Adam 学习率 0.005、50 epochs、随机种子 2026。沿用原 DistMult 打分、Xavier 初始化、margin=1 的排序损失和随机负尾实体采样；完整保留输入图谱中的边记录，不进行 seed 提取、黑名单过滤或 Top-k。

训练默认每 100 步打印当前 batch loss、本轮累计平均 loss 和批次进度，第一批、最后一批与中断时也输出。Notebook 中调整 `LOG_EVERY_STEPS`；命令行使用 `--log-every-steps`。日志频率独立于检查点保存频率，可以在续训时调整。

### 停止与恢复

按 **Ctrl+C**，或在 Linux 向进程发送 **SIGTERM**，训练会在当前优化步骤完成后保存并退出：

```bash
python clinicalKG/full_primekg.py --raw-kg kg.csv --epochs 50 --device cuda --resume
```

`--epochs` 表示总目标轮数；例如已经完成 12 轮，指定 50 会继续到第 50 轮。恢复时保持原图谱、batch size、嵌入维度、学习率与随机种子一致；脚本检查图谱 SHA-256 和索引对应关系。可在相同设置下提高总目标轮数。

每 1000 个训练步骤及每个 epoch 结束也保存，可用 `--save-every-steps 500` 调整频率。直接强制杀死进程时，使用最近一次已保存的检查点恢复。默认输出目录已存在检查点时，首次训练命令会提示加 `--resume`，防止覆盖。

### 生成文件

| 文件 | 用途 |
|---|---|
| `PrimeKG-Embeddings.npy` | 四队列共享的实体嵌入 |
| `PrimeKG-Entity2ID.json` | 与矩阵行严格对应的实体索引 |
| `PrimeKG-Relation2ID.json` | 关系索引 |
| `PrimeKG-DistMult-last.pt` | 模型、优化器、epoch、批次位置、当前打乱顺序、随机状态等续训信息 |
| `PrimeKG-LossHistory.json` | 已完成 epoch 的损失历史 |
| `PrimeKG-LossCurve.png` | 损失曲线 |
| `PrimeKG-TrainingState.json` | 图谱指纹、设置、进度和保存原因 |

`.pt` 和每个 JSON/NPY 文件均通过临时文件后替换写出。中断时导出的 `.npy` 对应当前保存权重；将同一次保存的嵌入及索引一起使用。云端训练后，将这些结果放回本机 `clinicalKG/shared/FullPrimeKG/` 即可，四队列自动共用。

继续沿用原项目的基于实体名称的索引方式，因此实体数量以脚本对输入 `kg.csv` 的实际统计为准。

## 4. PrimeKGL：每个队列只生成自己的患者映射，再训练分类模型

共享预训练完成后：

```bash
python clinicalKG/PrimeKGL/ADNI_PrimeKGL/ADNI_PrimeKGL_build.py --stage map
python clinicalKG/PrimeKGL/ADNI_PrimeKGL/ADNI_PrimeKGL_train.py
```

其余三个队列同样运行：

```bash
python clinicalKG/PrimeKGL/AIBL_PrimeKGL/AIBL_PrimeKGL_build.py --stage map
python clinicalKG/PrimeKGL/AIBL_PrimeKGL/AIBL_PrimeKGL_train.py
python clinicalKG/PrimeKGL/NACC_PrimeKGL/NACC_PrimeKGL_build.py --stage map
python clinicalKG/PrimeKGL/NACC_PrimeKGL/NACC_PrimeKGL_train.py
python clinicalKG/PrimeKGL/PPMI_PrimeKGL/PPMI_PrimeKGL_build.py --stage map
python clinicalKG/PrimeKGL/PPMI_PrimeKGL/PPMI_PrimeKGL_train.py
```

`--stage map` 保留原患者阈值/常模、序列与 mask 生成流程，输出各自的 `数据集-PrimeKG[-Raw]-Entities.json`。其中图谱标准化步骤输出共享的完整 `PrimeKG-G.csv`，不生成队列两跳子图。

分类 notebook 的原有资产变量名保留；副本中的路径解析器将各队列 `*-Embeddings.npy` 和 `*-Entity2ID.json` 引用映射到公共目录，将 `*-Entities.json` 引用映射到本队列目录。这里不会读取旧队列子图的历史嵌入。

每个 `数据集_PrimeKGL_build.py --stage kge` 都调用同一个公共 Full PrimeKG 训练器，保留这四个入口是为了使用方便，**不需要依次执行四遍**。NACC/PPMI 原 notebook 中集成的损失图功能也由公共训练器的绘图函数提供。

## 5. DemRKL：保留原 DemRKG 生成和训练

例如 ADNI：

```bash
python clinicalKG/DemRKL/ADNI_DemRKL/ADNI_DemRKL_build.py --stage all
python clinicalKG/DemRKL/ADNI_DemRKL/ADNI_DemRKL_train.py
```

其他队列将路径中的 `ADNI` 替换为 `AIBL`、`NACC`、`PPMI`。可分阶段执行 `--stage graph`、`--stage map`、`--stage kge`、`--stage plot`。AIBL 原额外数据分布检查保留在 notebook 中，命令入口只执行建图、映射、KGE 和独立绘图阶段。

DemRKL 原 k 参数保持：ADNI/NACC 为 5、AIBL/PPMI 为 6；原 KGE 轮数、患者映射阈值、模型结构、缺失值处理、五次训练与患者划分均沿用源代码。此次未加入新的消融或调参流程。

## 6. 下游结果与检查

分类脚本沿用五个训练种子 `11、23、37、51、67`，共享原患者级固定划分（约 64%/16%/20%），按验证损失选择模型，写出逐记录预测和汇总。

默认结果保存在 `clinicalKG/results/数据集/PrimeKG/` 或 `clinicalKG/results/数据集/DementiaHKG/`。可以通过 `KG_ABLATION_OUTPUT` 改变结果根目录。

检查任一入口而不训练：

```bash
python clinicalKG/PrimeKGL/ADNI_PrimeKGL/ADNI_PrimeKGL_build.py --stage kge --dry-run
python clinicalKG/DemRKL/ADNI_DemRKL/ADNI_DemRKL_train.py --dry-run
python -m unittest discover -s clinicalKG/tests -v
```

测试使用小型合成图，验证全边保留、Ctrl+C/SIGTERM 保存、批次中断续训与连续训练一致、共享嵌入路径、独立患者映射，以及复制后原文件保持不变；不运行完整 PrimeKG 或真实 MRI 训练。

历史验证记录（2026-10-26）：8 项合成/路径测试通过，命令行小图训练、恢复至更高目标 epoch 及损失图导出通过。使用 PyTorch 2.10.0 CPU；本机 Anaconda 的 NumPy/PyTorch 存在 OpenMP 重复加载问题，验证进程通过设置 `MKL_THREADING_LAYER=SEQUENTIAL` 使用串行 MKL，未修改训练代码或系统环境。当前运行状态与结果核验见 `record/` 中带日期的记录。
