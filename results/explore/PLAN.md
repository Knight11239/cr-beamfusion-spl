# 事后探索性分析：计划（用户 2026-11-14；在看到下列分析的任何结果前提交）

这些分析在事前登记的对照结果（`results/controls/REPORT.md`）公布之后提出，**全部是事后探索性分析**，与事前计划的确认性结果分开报告，
不进入 56 项检验族，不改写其判读。

1. **MIRD 完整对比**（`scripts/explore_mird_full.py`，只用已有逐样本 CSV）：G2-Affine-mm、G2-LearnW0-mm、G2-BeamFusion-mm、G2-BeamFusion、
   AF-DMA(4)、AF-DMA(5)、ACC(μ = 0.001)、DMA-GSC-MPDR（失配验证集选参，λ = 1、δ = 1e-3）；六个 T60–子阵条件；
   ΔSI-SDR（x_dp、x_ref）、STOI、PESQ（x_dp）、只含目标直达分量时的失真 SDR。单列计划中 M2（G2-LearnW0-mm − G2-BeamFusion-mm）的结果（取自 completed-001）。
2. **MIRD 上凸组合为何更好的诊断**（`scripts/explore_band_diag.py`，Kaggle CPU）：G2-Affine-mm 与 G2-BeamFusion-mm（另列 LearnW0-mm、BeamFusion 作参照）
   在 MIRD 与 G2 仿真上按频带的实测 WNG（中位数、低于 0 dB 与 −10 dB 的比例）、只含目标直达分量时的失真、输出误差及其目标 / 干扰 / 噪声分量。
   待检验的假说：“仿射滤波器的 WNG 更低，对实测阵列的未建模误差更敏感”。结论只写成假说，不作因果结论。
3. **WNG 下限敏感性**（推理时径向投影，不重新训练；`ckpt:...:proj_w0=W0`）：对 G2-Affine-mm 施加 W0 ∈ {−15, −10, −5, 0, 3} dB。
   - **唯一的确认性选择（事先固定）**：在 G2 **含失配验证集**（500 条，κ ~ U[0, 1]，增益 / 相位，每条样本固定种子；与含失配训练同分布）上，
     按 ΔSI-SDR（x_dp）的均值在这 5 个值中取最大者；并列时取较高的 W0（更强的约束）。未投影的原模型只作参照，不参加选择。
   - 选出的 W0 在 MIRD S1 / S2 上的结果为该选择的报告值；5 个 W0 在 MIRD 上的完整曲线**只作描述，不据此选参**。
