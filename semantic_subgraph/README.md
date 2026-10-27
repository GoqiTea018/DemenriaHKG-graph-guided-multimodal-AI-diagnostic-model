# DemRKG 语义子图

此文件夹集中存放新增的绘图脚本和 PNG。脚本只读取原有图谱 CSV，默认将图片写入本文件夹下的 `output/`，可从任意工作目录调用。

## 输入与依赖

从仓库根目录安装 `pip install matplotlib networkx`。脚本读取 `ADNI/`、`AIBL/`、`NACC/` 或 `PPMI/` 中的 `DementiaHKG/` 下的子图 CSV（ADNI 为 `DementiaHKG_PrimeKG_Sub.csv`，其余为 `数据集_DementiaHKG_PrimeKG_Sub.csv`）；这些图谱 CSV 需在本地准备。建图与训练入口见 [项目 README](../README.md) 和 [clinicalKG README](../clinicalKG/README.md)。

## 运行

在本文件夹打开终端，执行：

```powershell
python plot_semantic_subgraph.py
```

默认生成 `output/NACC-DemRKG-semantic-subgraph.png`，300 dpi，4800 × 3000 像素。采用与原概览图一致的白底、彩色圆形节点和浅灰连线，显示实体名称，并在右侧解释边上的关系编号。

需要其他数据集时，单独指定：

```powershell
python plot_semantic_subgraph.py --dataset ADNI
python plot_semantic_subgraph.py --dataset AIBL
python plot_semantic_subgraph.py --dataset PPMI
```

脚本使用现有 Python 环境中的 Matplotlib 和 NetworkX，不自动安装依赖或改动环境。

保留上一版分组示意风格，可指定输出文件名另行生成：

```powershell
python plot_semantic_subgraph.py --style schematic --output output/NACC-semantic-schematic.png
```

## 选图方式

- 默认围绕 Alzheimer disease、认知表型和 APP 选取可读的局部关系。
- 临床表型优先包含 Memory impairment、Cognitive impairment 和 Aphasia；解剖实体优先采用 cerebral cortex。
- 优先展示 APOE、MAPT 及其真实邻居；若某数据集未保留对应连接，选择该数据集中已存在的连接和蛋白。
- 默认网络图展示所选实体之间的全部关系类型，构成局部诱导子图。NACC 示例包含 12 个实体、20 条关系、8 类实体和 10 类关系。
- 每条连线都来自该数据集的 CSV。同一实体对、同一关系类型的镜像记录合并显示一次；边上的编号通过右侧说明对应易读名称及原始关系类型。
- 节点保留原始名称和类型，采用现有白底概览图的类型配色。网络布局固定随机种子，布局计算仅用于排版。
- `--style schematic` 使用上一版分组关系骨架，展示其中 11 条精选关系。
- 各数据集按各自的真实连接选图，节点和类型数量可以不同。PPMI 选项同样展示其图谱中的认知障碍相关局部连接。

论文正文建议先采用一个 NACC 语义示例，配合四个数据集的概览图和共享图例。需要比较队列间的具体知识差异时，再增加相应数据集的语义子图。
