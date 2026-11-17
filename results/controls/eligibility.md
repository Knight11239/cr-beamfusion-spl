# 新对照的训练前资格记录（research/analysis_plan_controls.md §2；用户 2026-11-13）

本记录在任何新对照训练开始前填写并提交；之后的修改须另写带日期的修订并说明当时是否已知结果。计划 v1.1 的比较矩阵、终点、阈值与判读规则不变。

## 1. 同头定义

| 项 | A（Affine-mm） | R（affine_real_mm） | C（convex_samehead_mm） | F（free_complex_mm） | 参照 B（BeamFusion-mm） |
| --- | --- | --- | --- | --- | --- |
| 骨干 | GTCRN（width 32、hidden 32、intra 双向，时间因果） | 同 | 同 | 同 | 同 |
| 最后一层 | ConvTranspose2d(32 → n_out, (1, 5), stride (1, 2), padding (0, 2))，含 bias，**无 BN**，恒等激活，接固定 ERB 合成 | 同 | 同 | 同 | 同，但**有 BN**（原 BeamFusion 头） |
| n_out 与映射 | 8：Re/Im w_p（p = 1..4）；c_D = 1 − Σw，c_p = w_p | 4：实数 w_p（p = 1..4），c 同 A | 5：logits（p = 0..4）→ softmax → c = α | 10：Re/Im w_p（p = 0..4，含 h_D）；c = w，不约束 Σc | 5：logits → softmax |
| 初始化 | 输出层权重 ×0.01、bias 0 | 同 A | 权重 ×0.01；bias b_0 = 0、b_p = −ln(4·0.95/0.05)（初始 α_D ≈ 0.95） | 权重 ×0.01；bias：Re w_D = 1，其余 0（初始 c ≈ e_D） | GTCRN 默认 |
| 有效系数 | Ẑ = Σ c_p Z_p，c 复数、Σc = 1 | c 实数、Σc = 1 | c 实数非负、Σc = 1 | c 复数、Σc 自由 | 同 C |
| 投影 / 后处理 | 无 | 无 | 无 | 无（不加投影、输出重标定或后滤波） | 无 |
| 精度 / 优化 | float32 训练；Adam 1e-3，验证损失 5 个 epoch 不降则减半，最低 1e-4；梯度经全部输出通道 | 同 | 同 | 同 | 同 |

所有输出通道都有效、都在梯度路径上（S 依赖每个通道）。**有效输出维数不同**（A 8、R 4、C 5、F 10）：因此 A、R、C、F 只称“头结构匹配”（同一层类型、无 BN、同一初始化尺度），不称逐参数完全相同。
实现：`src/models/gtcrn_fusion.py`（AffineFusionNet、SameHeadFusionNet）；配置 `configs/model_affine.yaml`、`model_affine_real.yaml`、`model_convex_samehead.yaml`、`model_free_complex.yaml`（除 name / kind / mode / alpha_d0 / weights 外逐项相同，已用 diff 核对）。

## 2. 初始功能（`scripts/controls_initial_function.py` → `initial_function.json`）

开发数据验证集 8 条（含失配验证同分布的增益 / 相位失配），f > 0，torch.manual_seed(1234) 下的初始化（实际训练的种子派生不同，见第 5 节）：

| 模型 | 可训练参数 | n_out | 最后 BN | abs(c_D − 1) 均值 / P95 | abs(c_p)（p ≥ 1）均值 / P95 | abs(Σc − 1) P95 | 初始输出相对 h_D 输出的偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 85,527 | 8 | 无 | 0.0065 / 0.0135 | 0.0037 / 0.0084 | 0.0000 | 0.024 |
| R | 84,883 | 4 | 无 | 0.0072 / 0.0191 | 0.0038 / 0.0099 | 0.0000 | 0.031 |
| C | 85,044 | 5 | 无 | 0.0499 / 0.0501 | 0.0125 / 0.0126 | 0.0000 | 0.032 |
| F | 85,849 | 10 | 无 | 0.0028 / 0.0057 | 0.0031 / 0.0071 | 0.0175 | 0.037 |
| B | 85,054 | 5 | 有 | 0.7934 / 0.8676 | 0.1984 / 0.3000 | 0.0000 | 0.567 |

A、R、C、F 的初始输出都在 h_D 输出的约 2–4% 以内；C 的 c_D 按设计约为 0.95（不正好在锚点）；F 的 Σc 初始偏离 1 约 1–2%（P95）；B 从接近均匀融合开始（偏差约 0.57），这是 B 与 C 的初始功能差别（C4 的混杂之一）。
**残留混杂**：初始功能不完全相同（C 约 0.95 而非 1；F 多一个自由复数自由度）；有效参数量与梯度参数化不同（softmax 对线性）。C1（A − C）不能解释为唯一因素的因果证明。

## 3. 共同训练协议

同一 x_dp 目标的 STFT MSE（`reference.train_target=direct`）；同一数据、字典与 STFT；含失配训练与含失配验证（`train_mismatch.enabled=true`、`train_mismatch.val=true`；κ ~ U[0, 1]，只含增益 / 相位，位置误差只用于测试）；seed_key = beamfusion，train.seed = 1234（种子 0）；micro-batch 15、有效 batch 30、50 个 epoch。
任务文件：`scripts/kaggle/jobs/gpu_ctrl_s2.json`（C + R）、`gpu_ctrl_s3.json`（F，以及 BeamFusion-mm 种子 1，train.seed = 1235）。A 为已有冻结权重 `results/checkpoints/stage4-train-b3/affine_mm_best.pt`（同一协议，gpu_stage4_b3）；B 为 `results/checkpoints/stage4-train-b5/beamfusion_direct_mm_best.pt`。

## 4. 检查点选择

只按预定验证损失（含失配验证集上的 STFT MSE）选 best.pt（train.py：每个 epoch 末验证损失创新低时保存）；不按测试指标挑 epoch，不从多次重启中挑选。若某次会话因预算中断，从 last.pt（含优化器、调度器、epoch、随机数状态）续跑到 50 个 epoch；实际完成的 epoch、best epoch、续跑日志与权重哈希随结果一起记录。未完成 50 个 epoch 的模型只作中期描述。

## 5. 共同输入

数据配置 `configs/data.yaml`（全局种子 20251028）；训练 RIR 为 crbf-rir-train-a/b/c，测试为 crbf-rir-eval / crbf-rir-kappa（Kaggle kernel 输出，版本在回收时记录于任务的 COLLECTED.json）；测试 ID 为 test_s1 / test_s2 各 600 条（κ = 0、1、2）。
训练数据顺序由 `derive_seed(train.seed, "train", seed_key)` 派生，各模型相同；失配增强由数据种子、样本序号与 epoch 派生，各模型相同。输出层形状不同会改变初始化随机数的消耗，因此**同一整数种子不等于逐参数初始化相同**；结论只针对给定的训练种子。

## 6. 算子资格（`tests/test_controls.py`，5 项通过）

- 对 A、R、C、F：网络输出 Ẑ 与 Σ_p c_p Z_p 一致，并与由系数重建的有效滤波器 h = Σ_p conj(c_p) h_p 作用于阵列信号的 hᴴy 一致（共轭约定；相对误差 < 1e-4）。
- 实系数（R、C）乘复谱后虚部保留；C 的 α 非负且和为 1。
- 输出层权重置零时，C 的系数正好为 [0.95, 0.0125 × 4]，F 的系数正好为 e_D（偏置初始化）。
- 评测：`src/evaluate.py` 对 samehead 模型直接取 coef（Ẑ = Σ c_p Z_p）；F 的实测 WNG 用一般式 abs(Σc)² / (vᴴKv)（`coef_wng_db`），不套用仿射式 1/(1/M + uᴴGu)。
- 下游评价定义（STFT、参考、DC 处理、指标）对所有模型相同；不对 F 单独加增益修正或后处理。

## 7. CPU 冒烟测试（开发数据）

convex_samehead_mm 与 free_complex_mm：训练 1 个 epoch、从 last.pt 续跑到第 2 个 epoch、用 best.pt 经 src.evaluate 与 scripts/eval_sets.py（含实测 WNG，κ = 1）评测，均正常（数值无意义，只验证流程）。

## 8. 文件哈希（sha256 前 16 位，本记录提交时）

| 文件 | sha256 |
| --- | --- |
| src/models/gtcrn_fusion.py | ea892da24aae2606 |
| src/models/gtcrn.py | 7fd827c993dedd75 |
| src/train.py | 2a1ccfb56a81cf1f |
| src/evaluate.py | 5dd4317cbe48e6b5 |
| configs/model_affine.yaml | ed2034403a44226d |
| configs/model_affine_real.yaml | cf1a4a372d2b019f |
| configs/model_convex_samehead.yaml | 258054cf3b643b29 |
| configs/model_free_complex.yaml | d626f7d772d57931 |
| configs/model_beamfusion.yaml | 512966fada4d5deb |
| configs/data.yaml | 415996816c7d2840 |
| scripts/kaggle/jobs/gpu_ctrl_s2.json | bef7300f320dff38 |
| scripts/kaggle/jobs/gpu_ctrl_s3.json | 5e562c3134a2e283 |
| results/checkpoints/stage4-train-b3/affine_mm_best.pt | 51c9b1a279bcd7dd |
| results/checkpoints/stage4-train-b5/beamfusion_direct_mm_best.pt | 1519e06e2324a12b |

注：gpu_ctrl_s2 / s3 的 max_hours 在提交时按剩余额度修改（不影响训练协议，只影响何时停下续跑），其哈希在提交后会变化，原因记在任务的 session_history。
