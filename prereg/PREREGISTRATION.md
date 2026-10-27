# CovProp 补强实验预注册（degradation-sweep protocol）

- 日期：2026-08-10（正式跑之前定稿；smoke 仅用于验证代码可运行，不用于调阈值）
- 背景：混池盲评（reading_room/venue_pool_ranking_kdd_icde.md，A4 = 9/11，lean reject / major revision）两大硬伤：
  1. 上游预测器差时（ETTm2）被朴素 recalibration 反超 >2×；
  2. 跨流的 share 统计不能预测干预增益（Spearman −0.29~−0.54 不显著、宽度口径 Pearson 显著为负），
     "observability primitive" 卖点兑现不足——运营者事前无法据其决定该不该用 CovProp。
- 修复路径（盲评原话）："给出可靠的事前适用域检测器，或收缩声明。"

## 目标

把 share/skill 统计从"描述性诊断"升级为**可验证的事前决策器**：
(a) 用系统性上游降质扫描划定适用域边界（何时用 CovProp、何时退回 recalibration），
(b) 用 label-free 的校准侧统计量构造干预选择 policy，量化其**决策增益**（相对盲选/全量单一干预）。

## 实验设计

### E1 上游降质扫描（适用域边界）

- 流：7 条真实上游误差流（NP, FR, ETTh1, weather, ETTh2, ETTm2, electricity），完全沿用 R3 协议
  （同 ridge 上游预测器、同划分、同 K、同 matured online 校准；不改任何 v1 冻结文件）。
- 降质机制：对上游预报注入白噪声
  `up_γ = up + γ · s_j · N(0,1)`，s_j = 该流校准段每通道干净上游误差 RMS；
  校准与测试一致注入（运营者记录的就是被部署的降质上游的误差——诚实设定）。
- 网格：γ ∈ {0, 0.5, 1, 2, 4}；seeds {0,1,2}。
- **机制 B（修正 1 增补）**：向持值预报混合，摧毁上游信息量而非仅加噪：
  `up_λ = (1−λ)·up + λ·persistence`，λ ∈ {0.25, 0.5, 0.75, 1.0}（λ=0 即 γ=0 点）。
  共 7×(5+4) = 63 个操作点（每点 3 seeds 均值）。
- 每个操作点记录：
  - label-free 校准侧统计：skill ratio `s = mean|up_γ−Xc|_cal / mean|persistence−Xc|_cal`（≥1 即不如持值），
    未截断聚合 share `ρ_cal`（1 − endo/half）；
  - 测试侧结果：oracle / naive / recal / covprop_mc / covprop_share 的 coverage、worst-tercile coverage、
    width、Winkler（协议与 R3 evaluate 完全一致）。

### E2 share/skill 驱动的干预选择 policy（决策增益）

在全部操作点上，比较以下选择器（每点从 {recal, covprop_mc, covprop_share} 三个干预中选一个）：

- **π_skill（主 policy，预注册阈值 τ=1.0）**：s ≥ 1.0 → recal；s < 1.0 → covprop_share。
- **π_skill+share（次 policy）**：s ≥ 1.0 或 ρ_cal < 0.05 → recal；否则 covprop_share。
- **π_backtest（修正 1 增补，主检测器）**：纯校准侧 backtest——把校准 B 段一分为二（B1 校准、B2 带标签评估，
  matured 协议不变），比较 {recal, covprop_mc, covprop_share} 在 B2 上的 Winkler，部署时选 backtest 赢家。
  完全事前（不看任何测试数据），是盲评要求的"可靠事前适用域检测器"的直接实现；share/skill 作为免标签廉价代理。
- 对照：always-recal、always-covprop_mc、always-covprop_share、
  random（三者均值期望）、oracle（每点事后 Winkler 最优——上界，非可部署）。

指标：每操作点 Winkler（3 seeds 均值）与 worst-tercile coverage；
决策增益 = 相对各 always-* 的平均 Winkler 改善（%），regret = 相对 oracle 的平均差（%）。

## 预注册假设与判据（跑前定死，不许挑格子）

- **H1（事前检测器有效性）**：1{s<1} 预测 "covprop_share 在该操作点 Winkler 优于 recal"
  的符号准确率 ≥ 0.80（全部操作点）；辅助报告 −s 的 AUC。
- **H2（决策增益）**：π_skill 的平均 Winkler 相对 oracle regret ≤ 3%，
  且相对 always-recal 与 always-covprop_share 均有改善（跨操作点配对 Wilcoxon p < 0.05）。
- **H3（边界形状）**：每条流内，covprop_share 相对 recal 的 Winkler 增益沿每条降质轴（γ 与 λ 分别）
  单调不增（逐流逐轴 Spearman ≤ 0，符号检验 p < 0.05）；存在符号翻转的流，其翻转点的 s 落在 [0.9, 1.1]。

- **H4（修正 1 增补，backtest 检测器）**：π_backtest 的平均 Winkler 相对 oracle regret ≤ 3%，
  且相对 always-recal 与 always-covprop_share 均有改善（配对 Wilcoxon p < 0.05）；
  其 backtest 赢家与测试侧赢家（Winkler 最优干预）的一致率 ≥ 0.80。

### 修正 1（2026-08-10，正式跑之前）

smoke（2 流 × γ∈{0,4} × 1 seed，仅验证代码可运行）暴露设计缺口：白噪声注入下 NP 在 skill=2.9 时
CovProp 仍胜 recal（诚实传播大噪声仍比均匀 recal 高效），说明 ETTm2 失效可能由"上游无信息"
而非"上游误差大"驱动，单一机制 A 无法划出边界。故在跑任何正式格点之前增补机制 B、π_backtest 与 H4；
未依据 smoke 调整任何已注册阈值（τ_skill=1.0、τ_share=0.05 不变）。
smoke 已提示 H1/H3（skill 阈值口径）可能失败——若失败将如实照登。

判据全部通过 → 卖点改写为"适用域内的决策增益"（v2 稿新目录）；
任一失败 → 如实照登负结果；若 H4 通过而 H1/H3 失败，卖点收缩为
"校准侧 backtest 是可靠事前检测器，share/skill 仅为流内诊断"；全部失败 → 降级/改道建议。

## 运行

- 代码：`experiments/covprop_sweep.py`（E1 网格）+ `experiments/policy_analysis.py`（E2 分析）。
- 纯 CPU；xu-1 上 `/home/dell/anaconda3/envs/tsf-research/bin/python`。
- smoke：`--smoke`（2 流 × γ∈{0,4} × 1 seed）；正式：全网格，输出 `results_v2/`（独立目录）。
- 复现：`bash run_v2.sh`。
