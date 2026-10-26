# CovProp R1 修复实验预注册（oral_loop/r1_fix）

- 日期：2026-08-11（正式跑之前定稿）
- 背景：独立审稿 R1（oral_loop/R1_review.md，6.9 weak accept）两条主修复线：
  - G1-5（W1）：系统开销与部署面未兑现——全文无运行时/存储开销数字；
  - G1-6（W2）：H4 backtest selector 验证域受限——无时间外推、无未退化域验证、
    sweep 仅 3 seeds、Wilcoxon 把同流 9 点当独立样本。
- 约束：frozen 目录零改动；raw 零改动；新输出写 oral_loop/r1_fix/results/。

## E3 系统开销微基准（bench_overhead.py）

- 流：7 条真实上游误差流（NP, FR, ETTh1, weather, ETTh2, ETTm2, electricity），
  R3 协议、未退化上游、seed 0，CPU（xu-1，单进程）。
- 测量（每流）：
  1. 批量测试路径 per-window 延迟与吞吐：recal（preds_any + online_q）、
     CQR（2×H pinball head 前向 + online_q）、CovProp-MC，K ∈ {8,16,32,64,128}
     （mc_quantiles_any + stage-2 online_q）、CovProp-Share 增量（share 分桶 + 逐桶 λ_b）。
     计时 time.perf_counter，重复 3 次取中位数。
  2. 单窗口在线延迟：n=1 的 covprop_mc（K=64）与 recal 前向，重复 20 次取中位数。
  3. 内存/存储：误差池字节数（eps 池 nbytes）、残差池字节数、MC 采样缓冲峰值
     （tracemalloc 包住 mc_quantiles_any）。
  4. 增量维护成本：每次 matured feedback 到达时更新 λ（split_q 重算，池大小 = 校准
     B 段规模）与 λ_b（分桶重算）的单步耗时。
- 报告口径：每流 JSON（overhead_&lt;stream&gt;.json）+ 汇总表（论文新增小节）。
- 性质：测量型实验，无接受/拒绝阈值；数字如实照登（包括不利的，如 K=64 相对
  recal 的倍数开销）。

## E4 H4 backtest selector 验证域扩展（backtest_extrap.py + policy_analysis_streamlevel.py）

沿用 icde2027_v2_prep 的 63 点降质网格（7 流 × [noise γ∈{0,.5,1,2,4}; blend
λ∈{.25,.5,.75,1}]），协议与 covprop_v2.py 完全一致，改动仅两处（见 E4a/E4c）。

### E4a 时间外推（决策点前移）
- 原 H4：backtest 在校准 B 段后半（B2a 校准 / B2b 评估，决策点紧邻测试段）。
- 新增 **early backtest**：决策窗口整体前移——校准于 B[0:q1]、带标签评估于
  B[q1:h2]（q1 = h2/2），全部决策信息止于 B 段中点，与部署段之间留出半个
  B 段（约 400 窗）的空档；部署段仍为原测试段。上游若在持续漂移，
  early 决策与部署的时间距离相对 late 版加倍。
- **H5（预注册判据）**：π_backtest-early 的平均 Winkler regret（vs 事后最优）≤ 3%，
  且与 late 版赢家一致率 ≥ 0.70。不达标则如实照登并在论文中收窄 H4 声明
  （"决策点需临近部署段"）。

### E4b 未退化域验证
- 网格中 γ=0 的 7 个未退化真实流操作点（5 seeds），报告 π_backtest 相对逐点
  事后最优干预的 regret 与选择分布。
- **H6（预注册判据）**：未退化 7 流上 mean regret ≤ 2%。评审原文以 12 数据集
  全 roster 为问；本轮以 7 条真实上游误差流为验证域（replay 合成腐蚀协议不含
  真实上游日志、backtest 所需的校准侧误差池语义不同），如实在论文限定。
- 选择菜单与 H4 相同（{recal, covprop_mc, covprop_share}，3 干预）。

### E4c 种子扩容 + 流级显著性
- 全 63 点 sweep 由 3 seeds 扩至 **5 seeds**（seeds {0,1,2,3,4}；0–2 与原结果
  逐点可复核一致）。
- 显著性主检验旁证：把 63 点按流聚类（n=7），
  (i) 流级配对符号检验（每流内先对 9 点取均值 regret 差）；
  (ii) 层次 bootstrap（先重采样流、再重采样流内点，B=10000）。
- **H7（预注册判据）**：π_backtest vs always-recal 的流级符号检验 7/7 同向
  （sign test p = 2^{-7}×2 ≈ 0.016）或层次 bootstrap 95% CI 不含 0。
  若不显著，如实照登并把论文中 p 值口径改为"点级 Wilcoxon（承认同流相关）+
  流级检验"双披露。

## 运行

- python：xu-1 /home/dell/anaconda3/envs/tsf-research/bin/python，CPU。
- E3：`python bench_overhead.py`（输出 results/overhead_*.json）。
- E4：`python backtest_extrap.py`（输出 results/sweep5_*.json，含 late/early 双 backtest）；
  `python policy_analysis_streamlevel.py --dir results`（输出 results/policy_r1fix.json）。
- 原 results_v2/ 不改动；seeds 0–2 的复核仅做只读对比。
