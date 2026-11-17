# 第19项验证记录

日期：2026-11-11。本次代码、清单和示例输出均位于 `research/control_analysis/`，冻结计划未修改。

- `python -m pytest -q research/control_analysis/test_controls.py`：**42 passed**（Python 3.12.14；SciPy 1.18.1；NumPy 2.5.4）。
- 实际命令：`python -m research.control_analysis.run --out research/control_analysis/example_empty_run`。退出码0，输入错误0；全部源码SHA与输出登记一致。第二次相同旧输入空跑的18份输出逐字节一致（bootstrap包含在内）。
- 固定主账本H8/HF/HG2为36/16/4，共56；全部“未执行”、eligible=0、展示p为空。次指标三族各34，共102；未缩小校正分母。
- 只读取显式固定的23个唯一文件：16份旧逐样本CSV、4份旧检查点（仅哈希）、2份旧MIRD元数据与冻结计划。未访问C/R/F/GBmm/GB输入；pending路径不访问的负面测试通过。
- 旧A−B与GL−GA另存54行SMOKE；数据配对全部完整，均不计入新主假设。A−B六格均值差、CI、r_rb和原始p与旧统计账本交叉核对通过。四格额外失配退化按600 ID四路配对核对通过。
- 六格A−B的T60分层bootstrap：20,000次；G2/MIRD的GL−GA仅作既有格式检查。MIRD S1七指标均标单轨迹不可估计、CI空；S2七指标均可计算，簇数见原始输出。
- TOST ±0.5边界、缺失槽Holm56、精确六条件p下界0.03125/Holm6下界0.1875、条件等权、无有限交集删样本、哈希/重复冲突、真实主输出NaN、共轭/iSTFT及冻结诊断来源/事件自洽均有测试。
- 合成矩阵、正弦与故障只用于数学测试，不构成任何新对照实验结果；没有训练、续跑、神经网络推理或新质量评测。

## 输入指标代理审查

旧数据最大绝对偏差（dB；固定容差1e−5，不自动证明输入协议资格）：

| 列 | 最大绝对偏差 |
| --- | ---: |
| `si_sdr_dp_in` | 6.28837060646e-07 |
| `si_sdr_in` | 7.09938138499e-07 |
| `snr_dp_in` | 3.7744828596e-07 |
| `snr_in` | 7.49474505568e-07 |


## 输出格式检查

| 文件 | 数据行数 |
| --- | ---: |
| `diagnostics_by_id.csv` | 0 |
| `diagnostics_summary.csv` | 12 |
| `f_target_descriptive.csv` | 12 |
| `legacy_smoke.csv` | 54 |
| `legacy_t60_bootstrap.csv` | 6 |
| `mird_condition_summary.csv` | 42 |
| `mird_six_conditions.csv` | 336 |
| `mird_trajectory_bootstrap.csv` | 56 |
| `paired_t60_bootstrap.csv` | 30 |
| `primary_56.csv` | 56 |
| `qualifications.csv` | 61 |
| `secondary_102.csv` | 102 |
| `t60_descriptive.csv` | 240 |


## 使用边界

既有CSV缺少来源合格的D诊断，故逐句诊断表为空；不能从SI-SDR倒推或把缺失视为约束通过。新评测需按README提供冻结混合系数的诊断CSV或显式NPZ索引，并提交资格证据；之后使用 `--manifest ... --allow-new-results --out ...` 一条命令生成全部计划输出。人工训练资格仍须作者核实，脚本只检查证据登记与哈希。
