# ADNI 与 AIBL：Full PrimeKG、DemRKG 与 CustomKG 运行结果分析

日期：2026-10-27。分析对象为 `clinicalKG/results_full_primekg/ADNI/` 与 `clinicalKG/results_full_primekg/AIBL/` 当前保存的结果。

更新说明：第 1—9 节保留 ADNI 分析；第 10—12 节补充 AIBL 的全部结果、掩码问题及修复记录，以及 PPMI 报错原因。AIBL 当前结果记录为“掩码修正前”；修正后的完整训练由用户自行运行，结果将写入 `clinicalKG/results_full_primekg/`。

## 1. ADNI 的主要发现

**DemRKG 以显著更紧凑的知识表示，取得了与完整 PrimeKG 非常接近的 ADNI 诊断表现；两种外部知识配置均优于当前 CustomKG 配置。**

- DemRKG 的 Accuracy、Precision、Recall、F1 和 AUC 均取得最高均值。
- 相比 Full PrimeKG，DemRKG 的 Accuracy、Macro-F1、AUC 分别高 0.24、0.26 和 0.13 个百分点；对应的校正后 p 值分别为 0.863、0.719、0.587，差值的 95% CI 均覆盖 0。这组结果的主要特点是性能接近。
- 相比 CustomKG，DemRKG 的 Accuracy、Macro-F1、AUC 分别提高 10.24、17.98 和 6.91 个百分点，三项比较经 Holm 校正后均达到 p<0.001。
- DemRKG 的边记录数减少 99.71%，实体数和实体嵌入文件大小均减少约 96.83%。因此，本批 ADNI 实验最适合突出“知识表示紧凑性与诊断表现”的结合。

## 2. 本次比较的设置与结果核验

| 项目 | 本次记录 |
|---|---|
| 诊断任务 | NC / MCI / AD 三分类 |
| 测试集 | 168 条记录，对应 168 名不同患者 |
| 测试类别分布 | NC 43；MCI 95；AD 30 |
| 数据划分 | 固定患者级分层划分，整体约 train/validation/test=64%/16%/20% |
| 重复方式 | 固定划分，训练种子 11、23、37、51、67，共五次独立训练 |
| 训练上限与选模 | 各次 200 epochs，按验证集最小 loss 选择 checkpoint |
| Precision、Recall、F1 | macro 平均 |
| 最终测试 AUC | macro one-vs-rest（OvR） |
| 统计单位 | 患者；同一患者的记录与五个种子的预测一起抽样/交换 |
| 统计重复次数 | bootstrap 10,000 次；配对置换 10,000 次 |
| 多重比较 | 每个数据集、每个指标内部，对 DemRKG vs CustomKG、DemRKG vs Full PrimeKG 两项比较作 Holm 校正 |

三种配置的运行记录具有相同的患者划分 SHA-256。重新读取 15 份测试预测文件，独立计算五项指标，结果与保存的 `summary.csv` 一致，最大绝对数值误差约为 1.11×10⁻¹⁶。15 次训练的验证 loss 最小值位置也均与保存的 `best_epoch` 一致。

本次还核对了运行记录中的嵌入、实体索引和患者查询文件指纹，均与当前文件一致。Full PrimeKG 的记录明确为 `full_original_graph`，包含 8,100,498 条原始边记录、129,262 个名称索引实体、30 类关系，嵌入维度为 128。该嵌入来自已完成 28 个 epoch、累计 445,356 步时保存的训练快照。

Full PrimeKG 与 DemRKG 使用相同的 128 维知识嵌入、下游模型结构、分类训练参数与训练/验证函数。837 个患者查询条目的有效实体集合逐项相同。两份患者查询 JSON 的文件哈希不同，但本次比较的有效实体支持一致。

## 3. 全部五项指标

以下均为百分比，格式为五次训练的均值±样本标准差。AUC 同样乘以 100 展示。

| 方法 | Accuracy | Precision（macro） | Recall（macro） | F1（macro） | AUC（macro OvR） |
|---|---:|---:|---:|---:|---:|
| CustomKG | 73.21±2.70 | 62.31±13.69 | 66.01±8.25 | 63.05±10.41 | 85.79±3.12 |
| Full PrimeKG | 83.21±1.85 | 81.93±2.22 | 79.87±1.84 | 80.77±1.87 | 92.57±0.49 |
| **DemRKG** | **83.45±1.60** | **81.95±1.77** | **80.37±2.06** | **81.03±1.83** | **92.69±0.27** |

DemRKG 与 Full PrimeKG 的测试均值和波动幅度相近。DemRKG 的 AUC 标准差为 0.27 个百分点，Full PrimeKG 为 0.49 个百分点；F1 标准差分别为 1.83 和 1.87 个百分点。CustomKG 的 F1 标准差达到 10.41 个百分点，反映了当前配置对训练随机种子的明显敏感性。

### F1 与 Macro-F1 的关系

本项目中表格原来的 **F1 列就是 Macro-F1**，代码调用为 `f1_score(..., average="macro")`。因此，将表头明确标为 F1（macro）只是补全计算口径，数值保持一致。

F1 是指标名称；Macro-F1 是多分类时的汇总方式。对本任务：

\[
F1_c=\frac{2P_cR_c}{P_c+R_c},\qquad
F1_{\mathrm{macro}}=\frac{F1_{\mathrm{NC}}+F1_{\mathrm{MCI}}+F1_{\mathrm{AD}}}{3}.
\]

即先计算各类别的 F1，再等权平均。Macro-Precision 和 Macro-Recall 也分别对各类别等权平均；Macro-F1 直接平均各类别的 F1，而非用已经平均后的 Precision 与 Recall 再计算调和平均。

本次测试集 MCI 占 95/168，约 56.55%，而 AD 只有 30 例。Macro 平均使三个类别获得相同权重，有助于同时呈现少数类别的识别表现。

## 4. 配对统计比较

差值及 95% CI 的单位为**百分点**，方向统一为 DemRKG−对照。最后一列是 Holm 校正后的 p 值。

| 对照 | 指标 | 均值差 | 95% CI | 校正后 p |
|---|---|---:|---:|---:|
| CustomKG | Accuracy | +10.24 | [5.48, 14.88] | 0.000200 |
| CustomKG | Macro-F1 | +17.98 | [13.25, 22.49] | 0.000200 |
| CustomKG | AUC | +6.91 | [4.31, 9.52] | 0.000400 |
| Full PrimeKG | Accuracy | +0.24 | [−1.07, 1.55] | 0.862714 |
| Full PrimeKG | Macro-F1 | +0.26 | [−1.20, 1.69] | 0.719128 |
| Full PrimeKG | AUC | +0.13 | [−0.20, 0.49] | 0.586541 |

与 CustomKG 的三项比较具有一致的正向效应，置信区间整体位于 0 以上。相对 Full PrimeKG 的差值较小，置信区间同时覆盖小幅正差异和小幅负差异。因此，现阶段可将 Full PrimeKG 与 DemRKG 描述为取得接近的观察性能，将显著差异的论述用于与当前 CustomKG 配置的比较。

## 5. 图谱规模与表示压缩

DemRKG 的规模由当前 `ADNI-DementiaHKG-G.csv` 实测得到，其实体集合与本次使用的嵌入索引一致。边数量按 CSV 中的**边记录/三元组行数**统计，与 Full PrimeKG 输入记录数对照。

| 项目 | Full PrimeKG | ADNI DemRKG | 减少比例 |
|---|---:|---:|---:|
| 边记录 | 8,100,498 | 23,630 | **99.71%** |
| 实体索引数 | 129,262 | 4,099 | **96.83%** |
| 关系类型 | 30 | 17 | — |
| 嵌入维度 | 128 | 128 | 相同 |
| 实体嵌入 NPY 文件 | 约 63.12 MiB | 约 2.00 MiB | **96.83%** |

这一组结果把论文的贡献落在了可测量的表示压缩上：领域精炼将完整生物医学图谱压缩到一个包含 4,099 个实体的队列相关知识表示，同时取得 Macro-F1 81.03%、AUC 92.69% 的诊断结果。

文件大小和图谱规模是本次直接测量的收益。知识预训练时间、GPU 峰值显存、下游推理耗时可以在后续实验中另行记录，形成独立的效率结果表。

## 6. 类别表现：主要差距来自哪里

下表为从测试预测重新计算的各类别 Recall，格式同样为百分比、五次均值±样本标准差。

| 方法 | NC Recall | MCI Recall | AD Recall |
|---|---:|---:|---:|
| CustomKG | 90.23±5.30 | 79.79±6.92 | 28.00±28.44 |
| Full PrimeKG | 85.58±2.55 | 87.37±2.35 | 66.67±4.71 |
| DemRKG | 88.37±3.29 | 86.74±1.41 | 66.00±2.79 |

CustomKG 的主要弱项是 AD 识别：种子 11 和 51 的 AD Recall 均为 0，五次平均约有 21.4/30 个 AD 样本被分为 MCI。Full PrimeKG 和 DemRKG 的对应数量分别约为 10.0 和 10.2，说明两种外部知识配置在 AD 与 MCI 区分上取得了明显改善。

DemRKG 相对 Full PrimeKG 的 NC Recall 高约 2.79 个百分点，MCI 与 AD Recall 分别低约 0.63 和 0.67 个百分点。这解释了两者整体指标非常接近、类别之间仍存在小幅权衡的现象。DemRKG 的 AD Recall 跨种子波动较小，本次为 2.79 个百分点。

## 7. 训练曲线与模型选择

![ADNI 三种图谱配置的训练和验证曲线](<C:/Users/HP/Design/DemenriaHKG graph-guided multimodal AI diagnostic model/clinicalKG/results_full_primekg/ADNI/training_curves.png>)

每行对应一种图谱，依次展示 Loss、Macro-AUC 和 Macro-F1；曲线为五次训练的均值，阴影为样本标准差。

| 配置 | 五次最佳 epoch | 平均初始验证 loss | 平均最佳验证 loss |
|---|---|---:|---:|
| CustomKG | 199、200、200、197、200 | 1.0961 | 0.6608 |
| Full PrimeKG | 200、200、196、195、195 | 1.0613 | 0.4017 |
| DemRKG | 198、189、191、195、194 | 1.0558 | 0.4017 |

Full PrimeKG 与 DemRKG 的 loss 下降、AUC 上升和 F1 改善轨迹整体接近，最终平均最佳验证 loss 均约为 0.4017。DemRKG 的最佳 epoch 多落在 189—198，Full PrimeKG 为 195—200。

CustomKG 的训练和验证指标改善较慢，接近第 200 轮时验证 loss 仍总体下降；它的较大种子波动同时出现在曲线和最终 F1 上。当前最佳模型集中在训练上限附近，后续可统一三种配置的验证集停止规则和训练预算，再观察稳定后的比较结果。

## 8. 结果解释与后续安排

### 本次最适合突出的研究结论

**领域精炼能够得到紧凑的诊断知识表示，在 ADNI 上取得接近完整 PrimeKG 的预测表现。** 同时，当前两种外部知识配置相比 CustomKG 的优势主要体现为更高的整体 F1、更好的 AD 识别和更稳定的跨种子结果。

可用于论文讨论的中文表述：

> 在 ADNI 三分类任务中，DemRKG 取得 83.45%±1.60% 的 Accuracy、81.03%±1.83% 的 Macro-F1 和 92.69%±0.27% 的 AUC，与完整 PrimeKG 的结果接近。领域精炼将图谱边记录数由 8,100,498 减少至 23,630，并将实体嵌入文件大小缩减约 96.83%，呈现了诊断知识表示的紧凑性。相较当前 CustomKG 配置，DemRKG 的 Accuracy、Macro-F1 与 AUC 均得到提高，患者级配对分析在 Holm 校正后均达到 p<0.001。

### 需要集中交代的实际比较条件

1. **CustomKG 与另外两种配置的知识维度不同。** 当前 CustomKG 嵌入为 1,091×32；Full PrimeKG 和 DemRKG 均为 128 维。因此，本次 CustomKG 对照反映的是现有整套配置的差异。若进一步拆解图谱构建本身的贡献，下一步优先增加统一嵌入维度的对照。
2. **本批实验是固定划分下的五次训练。** 当前置信区间与 p 值按这批测试患者及五次已训练模型计算。AIBL、NACC、PPMI 的完整新结果可以补充跨队列证据，患者级交叉验证可以补充划分稳健性。
3. **“性能接近”与正式非劣效结论的统计目标不同。** 当前 Full PrimeKG 对照的检验目标是差异是否偏离 0。若希望将论文主张进一步写成正式“保持性能/非劣效”，应预先给出有依据的性能差异界限，再按该目标设计分析。
4. **训练曲线与最终 AUC 的计算口径应分别标明。** 当前 ADNI epoch 函数中的训练 AUC 为 OvR、验证 AUC 为 OvO；最终测试表和配对统计均为 macro OvR。本报告的五指标表与统计表已由最终概率重新核验。后续展示曲线时可明确该口径，或者在新训练中统一记录。

建议的执行顺序是：先保存本次 Full PrimeKG 嵌入快照及指纹供四队列共用；完成其余队列；再补统一维度和构图消融。效率方面优先记录 KGE 训练时间与嵌入存储，使用实测数据补充紧凑性的实际收益。

## 9. 文件依据

- 五指标汇总：`clinicalKG/results_full_primekg/ADNI/metrics_mean_sd.csv`。
- 配对统计：`clinicalKG/results_full_primekg/ADNI/paired_statistics.csv` 与 `analysis_definition.json`。
- 每次训练：三种配置目录中的 `summary.csv`、`seed_11.csv` 至 `seed_67.csv`、`history_seed_*.csv`、`run_manifest.json`。
- 图谱规模：`ADNI/DementiaHKG/ADNI-DementiaHKG-G.csv`，以及本次运行记录引用的实体索引和嵌入文件。
- 本次重新计算的聚合核验数据：同目录 `ADNI_结果核验_2026-10-27.json`；该文件包含指标核验、类别召回率、平均混淆矩阵及训练诊断摘要。

## 10. AIBL：掩码修正前的运行记录

### 10.1 完成情况与数据核验

AIBL 的 CustomKG、Full PrimeKG 和 DemRKG 均已完成种子 11、23、37、51、67 的五次训练，每次记录 200 个 epoch。测试集包含 104 条记录、104 名患者，其中 NC 72、MCI 17、AD 15。三种配置的患者划分指纹一致，所有测试预测的患者、记录和标签逐项对齐。

从 15 份测试预测重新计算五项指标，与保存的 `summary.csv` 最大绝对误差为 5.55×10⁻¹⁷。各次最佳 epoch 均与记录中的验证 loss 最小值一致；运行记录中的知识资产哈希与当前文件一致。AIBL 使用与 ADNI 相同的 Full PrimeKG 嵌入快照，最终测试 AUC 为 macro OvO。

### 10.2 五项指标

以下为**掩码修正前**的保存值，单位为 %，格式为五次均值±样本标准差。

| 方法 | Accuracy | Precision（macro） | Recall（macro） | F1（macro） | AUC（macro OvO） |
|---|---:|---:|---:|---:|---:|
| CustomKG | 77.69±2.08 | 61.54±6.92 | 61.96±5.50 | 59.74±5.01 | 79.48±3.63 |
| Full PrimeKG | 85.96±1.29 | 80.45±2.85 | 77.44±1.89 | 78.48±2.06 | 86.65±0.74 |
| DemRKG | 85.96±1.29 | 80.45±2.85 | 77.44±1.89 | 78.48±2.06 | 86.65±0.74 |

CustomKG 的五次结果已完整保存。Full PrimeKG 与 DemRKG 的相同结果则触发了进一步代码核对：两者对应种子的测试概率、全部 epoch 历史以及模型状态张量均完全相同，概率的最大差值为 0。

### 10.3 已保存的配对统计

以下保留旧程序的统计输出供追溯。差值及 CI 单位为百分点，方向为 DemRKG−对照；p 值为同一指标内两项比较的 Holm 校正值。

| 对照 | 指标 | 均值差 | 95% CI | 校正后 p |
|---|---|---:|---:|---:|
| CustomKG | Macro-Precision | +18.91 | [10.17, 27.09] | 0.001000 |
| CustomKG | Macro-Recall | +15.48 | [7.68, 23.07] | 0.001400 |
| CustomKG | Macro-F1 | +18.74 | [10.17, 26.52] | 0.000800 |
| CustomKG | Accuracy | +8.27 | [4.04, 12.88] | 0.000800 |
| CustomKG | AUC | +7.16 | [3.48, 10.68] | 0.025197 |
| Full PrimeKG | Macro-Precision | 0.00 | [0.00, 0.00] | 1.000000 |
| Full PrimeKG | Macro-Recall | 0.00 | [0.00, 0.00] | 1.000000 |
| Full PrimeKG | Macro-F1 | 0.00 | [0.00, 0.00] | 1.000000 |
| Full PrimeKG | Accuracy | 0.00 | [0.00, 0.00] | 1.000000 |
| Full PrimeKG | AUC | 0.00 | [0.00, 0.00] | 1.000000 |

这些统计准确描述了修正前程序的预测差异。AIBL 的图谱效果比较将采用修正掩码后的重跑结果，原因见下节。

### 10.4 结果完全相同的原因与修复

知识序列构造函数以 `False` 标记有效实体、以 `True` 标记填充位，符合注意力层的 padding mask 约定。原模型在送入注意力层前执行：

```python
bool_kg_mask = (kg_mask == 0).bool()
```

这一步反转了掩码，使有效知识实体被屏蔽，而填充位被读取。AIBL 的 516 条患者记录中，315 条具有 11 个有效实体、201 条具有 12 个有效实体，序列长度均为 16。两种外部图谱配置的有效位置掩码完全相同，且均有剩余填充位；因此，在读取填充位的旧流程中，实际实体嵌入差异没有进入知识注意力。

使用 8 条真实 AIBL 知识序列进行修正前的前向核对，注意力分配给有效实体的总权重为 0，分配给填充位的总权重为 8；两种图谱输入对应的模型输出差值为 0。这一行为与五次训练结果完全相同的现象一致。

修复后直接使用输入布尔掩码，并为全填充序列设置零向量回退，避免注意力全部被屏蔽。AIBL 的三种 KG 配置统一采用修正后的模型副本；CustomKG 保留自身知识嵌入及维度。前向核对结果为：有效实体获得注意力、填充位权重为 0；改变填充向量不会改变输出，改变有效知识会改变输出；全填充序列的输出保持有限值。

因此，AIBL 当前表格作为修正前运行记录保存。CustomKG、Full PrimeKG 与 DemRKG 三种配置需要按相同修正重新训练后，再形成正式比较表。

### 10.5 类别表现、训练曲线与图谱规模

下表同样为修正前结果，单位为 %。

| 方法 | NC Recall | MCI Recall | AD Recall |
|---|---:|---:|---:|
| CustomKG | 92.22±3.49 | 17.65±7.20 | 76.00±17.38 |
| Full PrimeKG | 93.89±1.58 | 51.76±6.44 | 86.67±8.16 |
| DemRKG | 93.89±1.58 | 51.76±6.44 | 86.67±8.16 |

![AIBL 掩码修正前的训练和验证曲线](<C:/Users/HP/Design/DemenriaHKG graph-guided multimodal AI diagnostic model/clinicalKG/results_full_primekg/AIBL/training_curves.png>)

| 配置 | 五次最佳 epoch | 平均初始验证 loss | 平均最佳验证 loss |
|---|---|---:|---:|
| CustomKG | 195、200、200、198、199 | 1.1133 | 0.6597 |
| Full PrimeKG | 195、197、194、194、199 | 1.0913 | 0.4430 |
| DemRKG | 195、197、194、194、199 | 1.0913 | 0.4430 |

图中 Full PrimeKG 与 DemRKG 曲线完全重合，记录了掩码反转下的相同训练过程。CustomKG 的曲线改善较慢，MCI Recall 是该旧配置的主要弱项。这些曲线和类别统计可用于定位程序行为；修正后的曲线和类别结果由重跑生成。

图谱规模由实际图文件和嵌入文件独立测量，保留如下：

| 项目 | Full PrimeKG | AIBL DemRKG | 减少比例 |
|---|---:|---:|---:|
| 边记录 | 8,100,498 | 22,621 | 99.72% |
| 实体索引数 | 129,262 | 3,894 | 96.99% |
| 关系类型 | 30 | 18 | — |
| 嵌入维度 | 128 | 128 | 相同 |
| 实体嵌入 NPY 文件 | 63.12 MiB | 1.90 MiB | 96.99% |

AIBL CustomKG 的嵌入矩阵为 560×32；两种外部图谱配置均为 128 维。正式比较将结合修正后的分类表现与上述实测图谱规模。

## 11. 当前跨队列结论与结果状态

| 队列 | 当前状态 | 后续结果位置 |
|---|---|---|
| ADNI | 已完成五次训练与配对统计；分析见第 1—9 节 | `clinicalKG/results_full_primekg/ADNI/` |
| AIBL | 三种配置均有完整修正前结果；掩码已修正，等待用户重跑 | `clinicalKG/results_full_primekg/AIBL/` |
| PPMI | 输入检查报错已修复；同时修正相同的掩码问题，等待用户训练 | `clinicalKG/results_full_primekg/PPMI/` |
| NACC | 本次未提供运行结果 | `clinicalKG/results_full_primekg/NACC/` |

当前可报告的性能与紧凑性结合证据来自 ADNI。AIBL 已补充可复核的运行记录和独立图谱规模，修正后的训练将补充第二个队列的正式图谱对照。各队列的 Full PrimeKG 使用同一份原始图谱嵌入快照。

## 12. PPMI 报错修复与自行重跑步骤

### 12.1 PPMI 报错原因

主 notebook 的 CustomKG 训练 cell 调用 `check_inputs()` 时出现 `KeyError: 'filename'`。PPMI 的 CSV 文件在第二列保存 MRI 文件名，该列无表头；原 PPMI 训练代码使用 `row[1]`。主入口的输入检查曾固定读取名为 `filename` 的列，因而在训练启动前报错。

现已将 PPMI 的输入检查改为与原训练脚本一致地读取第二列。当前文件检查结果为 PD1 125 条、Control 132 条、Prodromal 80 条、SWEDD 72 条，四组 MRI 路径全部存在；模型实际使用的 PD 子集沿用原脚本与固定划分逻辑。三个 KG 配置的 notebook 和知识资产检查均通过。

PPMI 模型也存在与 AIBL 相同的掩码反转，现已同步修复。修复针对 `clinicalKG` 中的模型副本；AIBL/PPMI 的 CustomKG 训练也从这些副本加载模型，三种配置使用一致的掩码语义。原队列目录中的 `model1.py` 与复制时的 SHA-256 一致。

### 12.2 用户自行运行

1. 重新打开 `clinicalKG/Run_Experiments.ipynb`，先执行配置 cell。该单元自动定位整个主文件夹、重新加载 clinicalKG 模块，并将当前工作目录切换至主文件夹；从项目外启动时可填写 `MAIN_FOLDER`。
2. AIBL 设置 `DATASET_CHOICE = 3`；PPMI 设置 `DATASET_CHOICE = 4`。当前 notebook 保留用户选择的 PPMI，即 `4`。
3. 保持 `RUN_MODE = "all"`，先执行配置与输入确认单元，再由用户分别运行 CustomKG、Full PrimeKG、DemRKG 三个训练 cell。
4. 三种配置全部完成后，运行统计与五指标汇总 cell，最后运行训练曲线 cell。
5. AIBL/PPMI 修正后结果覆盖 `clinicalKG/results_full_primekg/` 中各自队列的原文件，文件名保持一致。每个训练 cell 重新训练对应配置的五个种子；全部完成后再生成统计与曲线。修正前聚合数值保留在本报告及核验 JSON 中。新运行记录包含模型源码哈希和掩码定义。

本次执行了预测指标重算、输入路径检查和固定输入的模型前向核对，未启动完整诊断训练。Full PrimeKG 的 DistMult 嵌入继续复用已保存文件。

### 12.3 新增核验依据

- AIBL 预测、统计与训练记录：`clinicalKG/results_full_primekg/AIBL/`。
- AIBL 修正前聚合核验：`clinicalKG/record/AIBL_结果核验_2026-10-27.json`，包含全部五指标、类别召回率、相同预测/权重核对及原掩码行为核对。
- AIBL/PPMI 修复后的前向与路径核验：`clinicalKG/record/AIBL_PPMI_掩码修复核验_2026-10-27.json`。
- 输入检查与运行记录：`clinicalKG/evaluation.py`；模型路径选择：`clinicalKG/experiments/config.py` 与 `clinicalKG/classification_worker.py`。

2026-10-27 入口更新：针对配置 cell 中旧内核缓存引起的 `ImportError: cannot import name default_output_root`，首个配置 cell 已加入模块缓存刷新，并直接定义原结果目录。已从主文件夹、clinicalKG、PPMI 及其嵌套脚本目录执行配置核对，均定位到同一主文件夹；PPMI 输入检查通过。本次未启动完整训练。
