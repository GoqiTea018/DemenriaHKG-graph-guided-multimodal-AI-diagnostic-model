# 图谱规模审计

2026-10-27 使用 [graph_overview.py](tools/graph_overview.py) 对完整输入 CSV 重新统计。Full PrimeKG 在四个队列间共享；四个 DemRKG 是各队列独立构建的子图。

## 规模总表

| 图谱 | 实体（名称索引） | 实体类型 | 边记录 | 去重有向三元组 | 关系类型 |
|---|---:|---:|---:|---:|---:|
| FullPrimeKG | 129,262 | 10 | 8,100,498 | 8,099,991 | 30 |
| ADNI | 4,099 | 8 | 23,630 | 23,630 | 17 |
| AIBL | 3,894 | 8 | 22,621 | 22,621 | 18 |
| NACC | 6,547 | 8 | 42,909 | 42,909 | 17 |
| PPMI | 2,497 | 8 | 13,050 | 13,050 | 17 |

实体采用当前 DistMult 的名称索引口径。Full PrimeKG 原始 `x_index/y_index` 共 129,375 个，名称索引合并同名节点后为 129,262 个；四个 DemRKG 的原始索引数与名称实体数相同。实体类型按照原 `Graph.ipynb` 的规则，优先使用 x 端首次出现的类型，再用 y 端类型补齐。

边记录保留所有 CSV 行；去重有向三元组保留关系类型和方向。下表的有向实体对将同一方向的多种关系合并。绘图中再以无向线段合并双向连接，完整关系可在 `*-view.json` 中查看。

| 图谱 | 去重有向实体对 | 展示实体 | 展示无向实体对 | 展示有向三元组 |
|---|---:|---:|---:|---:|
| FullPrimeKG | 8,098,805 | 1,417 | 12,714 | 25,433 |
| ADNI | 23,617 | 574 | 1,411 | 2,441 |
| AIBL | 22,610 | 559 | 1,420 | 2,434 |
| NACC | 42,878 | 584 | 1,487 | 2,555 |
| PPMI | 13,039 | 518 | 1,270 | 1,973 |

## 实体类型数量

| 类型 | Full PrimeKG | ADNI | AIBL | NACC | PPMI |
|---|---:|---:|---:|---:|---:|
| anatomy | 14,032 | 143 | 146 | 143 | 134 |
| biological_process | 28,642 | 465 | 543 | 475 | 296 |
| cellular_component | 4,171 | 103 | 117 | 105 | 68 |
| disease | 17,075 | 1,090 | 553 | 2,095 | 480 |
| drug | 7,956 | 0 | 0 | 0 | 0 |
| effect/phenotype | 15,302 | 230 | 157 | 346 | 168 |
| exposure | 755 | 0 | 0 | 0 | 0 |
| gene/protein | 27,671 | 1,950 | 2,245 | 3,262 | 1,287 |
| molecular_function | 11,169 | 75 | 82 | 78 | 60 |
| pathway | 2,489 | 43 | 51 | 43 | 4 |

## 各关系类型的边记录数

| 关系类型 | Full PrimeKG | ADNI | AIBL | NACC | PPMI |
|---|---:|---:|---:|---:|---:|
| anatomy_anatomy | 28,064 | 8 | 10 | 9 | 6 |
| anatomy_protein_absent | 39,774 | 34 | 50 | 87 | 20 |
| anatomy_protein_present | 3,036,406 | 7,518 | 9,443 | 14,930 | 3,487 |
| bioprocess_bioprocess | 105,772 | 149 | 220 | 161 | 113 |
| bioprocess_protein | 289,610 | 2,291 | 2,834 | 2,329 | 1,488 |
| cellcomp_cellcomp | 9,690 | 10 | 18 | 15 | 14 |
| cellcomp_protein | 166,804 | 733 | 909 | 906 | 457 |
| contraindication | 61,350 | 0 | 0 | 0 | 0 |
| disease_disease | 64,388 | 200 | 67 | 421 | 92 |
| disease_phenotype_negative | 2,386 | 57 | 36 | 102 | 39 |
| disease_phenotype_positive | 300,634 | 6,813 | 2,332 | 13,490 | 2,953 |
| disease_protein | 160,822 | 1,783 | 2,108 | 5,762 | 1,370 |
| drug_drug | 2,672,628 | 0 | 0 | 0 | 0 |
| drug_effect | 129,568 | 0 | 0 | 0 | 0 |
| drug_protein | 51,306 | 0 | 0 | 0 | 0 |
| exposure_bioprocess | 3,250 | 0 | 0 | 0 | 0 |
| exposure_cellcomp | 20 | 0 | 0 | 0 | 0 |
| exposure_disease | 4,608 | 0 | 0 | 0 | 0 |
| exposure_exposure | 4,140 | 0 | 0 | 0 | 0 |
| exposure_molfunc | 90 | 0 | 0 | 0 | 0 |
| exposure_protein | 2,424 | 0 | 0 | 0 | 0 |
| indication | 18,776 | 0 | 0 | 0 | 0 |
| molfunc_molfunc | 27,148 | 15 | 23 | 18 | 19 |
| molfunc_protein | 139,060 | 480 | 620 | 569 | 943 |
| off-label use | 5,136 | 0 | 0 | 0 | 0 |
| pathway_pathway | 5,070 | 0 | 1 | 0 | 0 |
| pathway_protein | 85,292 | 275 | 368 | 278 | 28 |
| phenotype_phenotype | 37,472 | 204 | 94 | 294 | 110 |
| phenotype_protein | 6,660 | 210 | 249 | 646 | 261 |
| protein_protein | 642,150 | 2,850 | 3,239 | 2,892 | 1,650 |

## 来源与复现

- **FullPrimeKG**：`ADNI/PrimeKG/kg.csv`；[完整统计 JSON](PrimeKGL/figures/FullPrimeKG-statistics.json)。SHA-256：`0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69`。
- **ADNI**：`ADNI/DementiaHKG/DementiaHKG_PrimeKG_Sub.csv`；[完整统计 JSON](DemRKL/figures/ADNI-statistics.json)。SHA-256：`f72adabe258af7c9076f38788dc36a74d6dc7043f37f6c9eed6f37eaaa05436d`。
- **AIBL**：`AIBL/DementiaHKG/AIBL_DementiaHKG_PrimeKG_Sub.csv`；[完整统计 JSON](DemRKL/figures/AIBL-statistics.json)。SHA-256：`70fb1c9a2dc1b8cc060a960d861a42cc9f82d5367a5696ade6c1e1d9e1979774`。
- **NACC**：`NACC/DementiaHKG/NACC_DementiaHKG_PrimeKG_Sub.csv`；[完整统计 JSON](DemRKL/figures/NACC-statistics.json)。SHA-256：`9654ea2ebf76a636696c3cd126b1c0e73df89b3bafcd065260e5701374742627`。
- **PPMI**：`PPMI/DementiaHKG/PPMI_DementiaHKG_PrimeKG_Sub.csv`；[完整统计 JSON](DemRKL/figures/PPMI-statistics.json)。SHA-256：`5b9f609fb59ca76a7ab81bdcf9ec5e047c7f2f5acef37b0a93208a5f5a0af6a3`。

```bash
pip install -r clinicalKG/requirements.txt
python clinicalKG/tools/graph_overview.py
```

脚本按 250,000 行分块读取，使用稀疏邻接矩阵计算连通分量和最短路径，只对展示子图运行布局。每次生成 PNG、SVG、完整规模 JSON，以及含实体名称、类型和真实关系的展示子图 JSON；重新生成后据 JSON 更新本页与两份 README 的表格。
