"""Generate the dataset-selector / classification / statistics / curves notebook."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    ast.parse(text)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md('''# 四队列主入口：诊断训练、p 值与训练曲线

只改下面的 `DATASET_CHOICE` 一行选择数据集：**1=ADNI，2=NACC，3=AIBL，4=PPMI**，默认 ADNI。

这里的 **PrimeKG 明确定义为完整原始图谱**：保留所有输入边记录、实体类型和关系类型，不做 seed 邻域提取、黑名单过滤或 Top-k，四队列共用已保存的 DistMult 嵌入与索引。DemRKG 使用各队列领域精炼图谱，CustomKG 沿用原配置。

主入口消费已保存的图谱嵌入，三种配置各有一个独立的**下游诊断训练 cell**，每种配置运行五次。每轮实时打印训练/验证 Loss、Accuracy、Precision、Recall、F1、AUC（Precision/Recall/F1 为 macro），以及当前种子、epoch 和最佳 epoch。后面单独计算统计结果，最后一个代码单元格展示训练/验证曲线。

运行时保留整个 `clinicalKG/`，以及主目录中所选队列的 CSV、MRI 和原 CustomKG notebook。所有队列的结果均写入 `clinicalKG/results_full_primekg/`，沿用原文件名。重新运行某个 KG 的训练 cell 会覆盖该配置原来的五次训练结果；全部完成后重新生成统计表和曲线。
''')
md('''## 1. 选择数据集

通常只改第一行。首次 `RUN_MODE="all"`：可分别运行下面的三个 KG 训练 cell，全部完成后运行统计 cell。已有预测后用 `"stats"` 只计算统计，`"check"` 只检查输入，`"train"` 只训练。五次种子为 11、23、37、51、67，采用同一患者级固定划分。
''')
code('''DATASET_CHOICE = 1  # 1=ADNI；2=NACC；3=AIBL；4=PPMI

RUN_MODE = "all"     # all / stats / check / train
METRICS = ("precision", "recall", "f1", "accuracy", "auc")
RESAMPLES = 10000   # bootstrap 与置换各 10000 次
MAIN_FOLDER = None  # 自动定位整个主文件夹；需要手动指定时填写主文件夹路径

from pathlib import Path
import importlib
import os
import sys
_cwd = Path.cwd().resolve()
_starts = (Path(MAIN_FOLDER).expanduser().resolve(),) if MAIN_FOLDER else (_cwd, *_cwd.parents)
CLINICALKG_ROOT = next((p for start in _starts for p in (start / "clinicalKG", start)
                       if (p / "evaluation.py").is_file() and (p / "paths.py").is_file()), None)
if CLINICALKG_ROOT is None:
    raise FileNotFoundError("找不到 clinicalKG，请将 MAIN_FOLDER 指向包含 clinicalKG、ADNI、AIBL、NACC、PPMI 的主文件夹。")
if MAIN_FOLDER:
    os.environ["CLINICALKG_PROJECT_ROOT"] = str(CLINICALKG_ROOT.parent)
_package_parent = str(CLINICALKG_ROOT.parent)
sys.path[:] = [_package_parent] + [p for p in sys.path if p != _package_parent]
# Reload project modules when rerunning this cell after a source update.
importlib.invalidate_caches()
for _module_name in tuple(sys.modules):
    if _module_name == "clinicalKG" or _module_name.startswith("clinicalKG."):
        del sys.modules[_module_name]
from clinicalKG.evaluation import (
    DATASET_CHOICES, PRIMEKG_DEFINITION, LABELS,
    train_classifiers, check_inputs, run_statistics, metric_table,
    inspect_full_primekg, plot_training_curves,
)
from clinicalKG.paths import PROJECT_ROOT, full_primekg_dir, asset_path
from clinicalKG.experiments.config import GRAPHS, asset_names
os.chdir(PROJECT_ROOT)

if DATASET_CHOICE not in DATASET_CHOICES:
    raise ValueError("请选择 1=ADNI、2=NACC、3=AIBL、4=PPMI")
if RUN_MODE not in ("all", "stats", "check", "train"):
    raise ValueError("RUN_MODE 请选择 all、stats、check 或 train")
DATASET = DATASET_CHOICES[DATASET_CHOICE]
OUTPUT_ROOT = CLINICALKG_ROOT / "results_full_primekg"
print("所选数据集：", DATASET)
print("运行方式：", RUN_MODE)
print("主目录：", PROJECT_ROOT)
print("当前工作目录：", Path.cwd())
print("结果目录：", OUTPUT_ROOT)
print("PrimeKG 定义：", PRIMEKG_DEFINITION)
''')
md('''## 2. 确认输入

Full PrimeKG 自动读取 `clinicalKG/shared/FullPrimeKG/` 或你下载到 `clinicalKG/PrimeKGL/` 的文件。各队列的患者实体序列分别读取。PPMI 的 MRI 文件名按原脚本读取 CSV 第二列（无列名）。`stats` 模式只读取保存的预测及运行记录。
''')
code('''import pandas as pd
from IPython.display import display
if RUN_MODE != "stats":
    full_primekg_info = inspect_full_primekg()
    print("Full PrimeKG：", full_primekg_dir())
    print("边记录：", full_primekg_info["records"],
          "实体：", full_primekg_info["entities"], "关系类型：", full_primekg_info["relations"])
    print("KGE 已完成 epoch：", full_primekg_info["completed_kge_epochs"])
    input_rows = []
    for graph in GRAPHS:
        for role, filename in asset_names(DATASET, graph).items():
            path = Path(asset_path(DATASET, graph, filename))
            input_rows.append({"method": LABELS[graph], "asset": role,
                               "exists": path.is_file(), "path": str(path)})
    display(pd.DataFrame(input_rows))
else:
    print("统计模式：读取", OUTPUT_ROOT / DATASET)
''')
md('''## 3. 三种 KG 分别训练

下面三个代码 cell 相互独立，每个只运行对应的一种 KG。可以先运行任何一个，完成后再运行其余 cell；重新运行会覆盖对应配置的原结果。每个 cell 实时展示每轮训练/验证 Loss、Accuracy、Precision、Recall、F1、AUC（Precision/Recall/F1 为 macro）、种子与最佳 epoch，并保存预测、权重及曲线记录。

### 3.1 CustomKG：单独训练
''')
code('''if RUN_MODE in ("all", "train"):
    train_classifiers(DATASET, OUTPUT_ROOT, graphs=("CustomKG",))
elif RUN_MODE == "check":
    check_inputs(DATASET, graphs=("CustomKG",))
else:
    print("统计模式：跳过 CustomKG 训练。")
''')
md('''### 3.2 PrimeKG：完整原始图谱，单独训练

直接使用已保存的 Full PrimeKG 嵌入，运行所选队列的五次诊断训练。
''')
code('''if RUN_MODE in ("all", "train"):
    train_classifiers(DATASET, OUTPUT_ROOT, graphs=("PrimeKG",))
elif RUN_MODE == "check":
    check_inputs(DATASET, graphs=("PrimeKG",))
else:
    print("统计模式：跳过 Full PrimeKG 诊断训练。")
''')
md('''### 3.3 DemRKG：单独训练
''')
code('''if RUN_MODE in ("all", "train"):
    train_classifiers(DATASET, OUTPUT_ROOT, graphs=("DementiaHKG",))
elif RUN_MODE == "check":
    check_inputs(DATASET, graphs=("DementiaHKG",))
else:
    print("统计模式：跳过 DemRKG 训练。")
''')
md('''## 4. 三种配置完成后，单独计算统计结果

差值为 **DemRKG − 对照**。95% CI 来自患者级配对 bootstrap，p 值为患者级双侧配对置换；同一患者的所有记录和五个种子一起抽样/交换。Holm 校正覆盖同一数据集、同一指标的两项对照比较。

旧 `experiments/results/` 保持原样，新结果记录 Full PrimeKG 资产指纹与划分。统计模式只接受带本入口运行记录的结果。
''')
code('''if RUN_MODE in ("all", "stats"):
    pipeline_result = run_statistics(
        OUTPUT_ROOT, DATASET, metrics=METRICS, resamples=RESAMPLES,
    )
elif RUN_MODE == "train":
    pipeline_result = (metric_table(OUTPUT_ROOT, DATASET), None)
else:
    pipeline_result = None
    print("检查模式：跳过统计。")
print("结果目录：", OUTPUT_ROOT / DATASET)
''')
md('''## 5. 性能与统计结果

性能汇总输出五项原始指标：Macro-Precision、Macro-Recall、Macro-F1、Accuracy 和 AUC；五项指标均计算 DemRKG 与两个基线的配对差异、95% CI 和 p 值。AUC 沿用各队列原计算口径（AIBL 为 OvO，其余为 OvR）。CSV 保留未格式化的数值。
''')
code('''if RUN_MODE != "check":
    performance, statistics = pipeline_result
    readable = performance[["method", "runs"]].copy()
    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
        column_label = {"precision": "Precision (macro)", "recall": "Recall (macro)", "f1": "F1 (macro)", "accuracy": "Accuracy", "auc": "AUC"}[metric]
        readable[column_label] = [f"{mean:.4f} ± {sd:.4f}" for mean, sd in
                            zip(performance[metric + "_mean"], performance[metric + "_sd"])]
    display(readable)
    if statistics is not None:
        display(statistics[["metric", "proposed", "baseline", "difference",
                            "ci_low", "ci_high", "p_value", "holm_p", "units"]])
    print("指标表：", OUTPUT_ROOT / DATASET / "metrics_mean_sd.csv")
    if statistics is not None:
        print("统计表：", OUTPUT_ROOT / DATASET / "paired_statistics.csv")
else:
    print("将 RUN_MODE 改为 all 后执行诊断训练与统计。")
''')
md('''## 6. 训练曲线（最后一个单元格）

每行一种图谱配置；三列为 Loss、Macro-AUC、Macro-F1。蓝色为训练集，橙色为验证集，阴影为五次训练的样本标准差。图片保存为 `training_curves.png`。

曲线读取新训练阶段保存的 `history_seed_*.csv`，重新打开 notebook 后也能单独查看。
''')
code('''if RUN_MODE == "check":
    print("当前为检查模式，尚未生成诊断训练曲线。")
else:
    training_figure = plot_training_curves(OUTPUT_ROOT, DATASET)
''')
for index, cell in enumerate(cells):
    cell["id"] = f"clinicalkg-evaluation-{index:02d}"
nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
path = ROOT / "Run_Experiments.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
