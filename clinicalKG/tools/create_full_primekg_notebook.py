"""Build a standalone Jupyter notebook from the verified Full PrimeKG trainer."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "full_primekg.py").read_text(encoding="utf-8")
tree = ast.parse(source)
definitions = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                            and node.name != "main")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    ast.parse(text)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    markdown('''# Full PrimeKG：一次训练，四个队列共享

这是**可独立运行**的 notebook，已内置 DistMult 与中断保存代码。云服务器上只需此 `.ipynb`、完整原始 `kg.csv`，以及 Python 环境中的 `torch`、`numpy`、`pandas`、`matplotlib`。

1. 将 notebook 与 `kg.csv` 放在同一个云端工作目录，打开后选择已安装 PyTorch 的 Python kernel。
2. 首次训练保持 `RESUME = False`，从上到下依次运行。
3. 停止时使用 Jupyter 的 **Interrupt Kernel / 中断内核** 按钮；当前优化步骤结束后会保存检查点与嵌入，等待显示 `Saved (interrupted)`。
4. 继续训练时将 `RESUME = True`，重新运行参数单元格和训练单元格；如果重启过 kernel，则从上到下运行。`EPOCHS` 是累计目标轮数。

在本项目中运行时自动使用主目录及共享输出目录。单独上传时默认在当前工作目录读取图谱，输出到 `clinicalKG/shared/FullPrimeKG/`。训练结束后，将共享目录中的结果文件一起下载回本机同名目录。

四个队列共用本 notebook 生成的嵌入及索引；每个队列的患者映射和诊断训练仍分别运行。
'''),
    markdown("## 1. 参数与相对路径\n\n一般只需确认图谱路径和训练参数。首次运行 `RESUME=False`，恢复时改为 `True`。\n"),
    code('''from pathlib import Path
import os

# None：自动识别项目主目录；单独上传时使用当前工作目录。
# 也可以填写相对于当前工作目录的路径，例如 ".." 或 "my_project"。
MAIN_FOLDER = None
RAW_KG_FILE = "kg.csv"       # 相对于主目录，也可以填写完整路径。
RESUME = False               # 首次 False；中断后继续训练改成 True。
EPOCHS = 50                  # 累计目标 epoch，不是新增 epoch 数。
BATCH_SIZE = 512
EMBED_DIM = 128
LEARNING_RATE = 0.005
SEED = 2026
SAVE_EVERY_STEPS = 1000      # 同时在每个 epoch 结束和正常中断时保存。
LOG_EVERY_STEPS = 100        # 每隔多少步打印当前 batch loss 和本轮平均 loss。

_cwd = Path.cwd().resolve()
if MAIN_FOLDER is not None:
    PROJECT_ROOT = Path(MAIN_FOLDER).expanduser().resolve()
else:
    _package = next((p for start in (_cwd / "clinicalKG", _cwd)
                     for p in (start, *start.parents)
                     if (p / "full_primekg.py").is_file() and (p / "paths.py").is_file()), None)
    PROJECT_ROOT = _package.parent if _package is not None else _cwd

RAW_KG_PATH = Path(RAW_KG_FILE).expanduser()
if not RAW_KG_PATH.is_absolute():
    RAW_KG_PATH = PROJECT_ROOT / RAW_KG_PATH
# 在原本地项目中复用已有的完整 kg.csv；云端优先读取上面指定的文件。
if RAW_KG_FILE == "kg.csv" and not RAW_KG_PATH.is_file():
    _candidates = [PROJECT_ROOT / "clinicalKG" / "shared" / "FullPrimeKG" / "kg.csv"]
    _candidates += [PROJECT_ROOT / d / kind / "kg.csv"
                    for d in ("ADNI", "AIBL", "NACC", "PPMI")
                    for kind in ("PrimeKG", "DementiaHKG")]
    RAW_KG_PATH = next((p for p in _candidates if p.is_file()), RAW_KG_PATH)
OUTPUT_DIR = PROJECT_ROOT / "clinicalKG" / "shared" / "FullPrimeKG"
SHARED_PRIMEKG = OUTPUT_DIR

def raw_primekg_path():
    return RAW_KG_PATH

print("主目录：", PROJECT_ROOT)
print("完整 PrimeKG：", RAW_KG_PATH)
print("共享输出目录：", OUTPUT_DIR)
print("恢复训练：", RESUME)
'''),
    markdown("## 2. 检查依赖、图谱表头和训练设备\n\n这一单元格只读取表头和少量示例，不开始训练。\n"),
    code('''import hashlib
import json
import random
import signal
import threading

import torch
from torch import nn
import numpy as np
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not RAW_KG_PATH.is_file():
    raise FileNotFoundError(f"请上传完整 kg.csv，或修改 RAW_KG_FILE：{RAW_KG_PATH}")
columns = set(pd.read_csv(RAW_KG_PATH, nrows=0).columns)
if not ({"x_name", "relation", "y_name"} <= columns
        or {"head", "relation", "tail"} <= columns):
    raise ValueError("图谱需包含 x_name/relation/y_name 或 head/relation/tail")
print("PyTorch：", torch.__version__)
print("训练设备：", DEVICE)
if DEVICE == "cuda":
    print("GPU：", torch.cuda.get_device_name(0))
print("图谱大小：", round(RAW_KG_PATH.stat().st_size / 1024**3, 3), "GiB")
print("检查点已存在：", (OUTPUT_DIR / "PrimeKG-DistMult-last.pt").is_file())
display(pd.read_csv(RAW_KG_PATH, nrows=3))
'''),
    markdown('''## 3. DistMult 与中断保存代码

运行本单元格定义训练函数。这里内置了与 `full_primekg.py` 相同的模型、目标函数和保存逻辑，无需另外上传 `.py`。

使用完整输入图谱，保留全部边记录，不执行 seed 提取、黑名单过滤或 Top-k。沿用原项目的实体名称索引方式。检查点保存模型、优化器、当前 epoch、批次顺序、批次位置和随机状态；恢复时校验输入图谱及超参数。
'''),
    code(definitions + "\n"),
    markdown('''## 4. 开始训练 / 恢复训练

运行后会先读取完整图谱并计算 SHA-256，再进入训练。默认每 100 步打印一次当前 batch loss、本轮平均 loss 和批次进度；第一批、最后一批和中断时也输出。已有检查点时请将第一部分的 `RESUME` 改为 `True`，保持 batch size、维度、学习率和随机种子一致。恢复时可提高累计目标 `EPOCHS`；日志频率 `LOG_EVERY_STEPS` 可以单独调整。

中断使用工具栏的中断按钮；直接关闭或强制重启 kernel 时，下一次从最近已保存的检查点恢复。
'''),
    code('''training_result = train_full_primekg(
    raw_kg=RAW_KG_PATH,
    output_dir=OUTPUT_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    embed_dim=EMBED_DIM,
    lr=LEARNING_RATE,
    device=DEVICE,
    seed=SEED,
    resume=RESUME,
    save_every_steps=SAVE_EVERY_STEPS,
    log_every_steps=LOG_EVERY_STEPS,
)
plot_loss(OUTPUT_DIR)
print("已中断并保存，可续训。" if training_result["interrupted"] else "训练完成。")
print("已完成 epoch：", training_result["completed_epochs"])
print("训练步数：", training_result["global_step"])
print("保存位置：", OUTPUT_DIR)
'''),
    markdown('''## 5. 查看保存结果

四队列共享 `PrimeKG-Embeddings.npy`、`PrimeKG-Entity2ID.json`、`PrimeKG-Relation2ID.json`。续训同时保留 `PrimeKG-DistMult-last.pt`。本单元格只查看现有文件，也可在中断后单独运行。
'''),
    code('''from IPython.display import Image, display

for path in sorted(OUTPUT_DIR.glob("PrimeKG-*")):
    if path.is_file() and not path.name.endswith(".tmp"):
        print(f"{path.name:36s} {path.stat().st_size / 1024**2:10.2f} MiB")
state_path = OUTPUT_DIR / "PrimeKG-TrainingState.json"
if state_path.is_file():
    saved_progress = json.loads(state_path.read_text(encoding="utf-8"))
    print("保存原因：", saved_progress["reason"])
    print("已完成 epoch：", saved_progress["completed_epochs"])
    print("当前 epoch 下一条记录位置：", saved_progress["next_row"])
embedding_path = OUTPUT_DIR / "PrimeKG-Embeddings.npy"
if embedding_path.is_file():
    embedding = np.load(embedding_path, mmap_mode="r")
    print("共享嵌入 shape：", embedding.shape)
    del embedding
curve_path = OUTPUT_DIR / "PrimeKG-LossCurve.png"
if curve_path.is_file():
    display(Image(filename=str(curve_path)))
'''),
]
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"full-primekg-{index:02d}"
destination = ROOT / "Full_PrimeKG.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
