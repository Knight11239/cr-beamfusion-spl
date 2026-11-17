# 第19项：事前对照分析脚本

实现依据：[冻结计划 v1.1](../analysis_plan_controls.md)，SHA-256 `db29ea7dadd364861a34fd4c6d03495544d82ce60708e27597daf31203d1d6c5`，不修改计划、训练代码、模型或原始结果。默认输入仅为763419c快照前已经存在的A/B和G2 Affine/LearnW0结果；**不发现、不探测、不载入C/R/F、G2 BeamFusion续跑或部分训练的结果。** 本轮不训练、不运行任何模型质量评测。

## 一条命令

仓库根目录、已有CPU Python环境（numpy/scipy；对应版本见旧审计requirements）：

```bash
python -m research.control_analysis.run
```

默认写入新的 `research/control_analysis/runs/legacy-smoke/`。不覆盖任何已有非空输出目录；再次运行请指定新的 `--out research/control_analysis/runs/another-run`。本次提交的实际空跑产物在 [example_empty_run/REPORT.md](example_empty_run/REPORT.md)，测试与来源核对见 [VALIDATION.md](VALIDATION.md)。

新模型评测完成后，**复制** `legacy_inputs.json` 为自己的已完成输入清单，显式填入新来源、method、检查点/文件SHA、诊断和资格证据，再执行：

```bash
python -m research.control_analysis.run \
  --manifest research/control_analysis/completed_inputs.json \
  --allow-new-results \
  --out research/control_analysis/runs/completed-001
```

这是分析已有评测/系数记录的命令，不是训练或神经网络推理命令。不要编辑被固定的 `legacy_inputs.json` 来自动吸收最新结果。默认模式只接受其未改变的A/B/GA/GL输入定义；任意显式新输入均需opt-in。pending模型的CSV、权重、诊断和资格证据路径连存在性都不检查。没有glob、最新文件/最佳测试分数自动选择、别名猜测或以旧模型代替新槽的逻辑。

## 模型键与输入清单

| 键 | 含义 | 允许数据集 | 本次 |
| --- | --- | --- | --- |
| A / B | Affine-mm / BeamFusion-mm | `8mic` | 旧文件、已启用 |
| C / R / F | 同头凸组合 / 实仿射 / 自由复融合 | `8mic` | pending，未读取 |
| GA / GL | 既有G2 Affine-mm / LearnW0-mm | `G2sim`, `MIRD` | 旧文件、已启用 |
| GBmm / GB | G2 BeamFusion-mm / BeamFusion续跑 | `G2sim`, `MIRD` | pending，未读取；不能用旧部分训练权重顶替 |

清单的 `state` 是 `existing`（仅旧模型）、`pending` 或 `evaluated`（未来显式评测完成）。新槽不能伪装为existing。8麦键不能提供MIRD/4麦输入。每个启用模型需要 `method`、`n_mics`、`seed=0`、`checkpoint:{path,sha256}` 和 `sources:[{dataset,path,sha256}]`。所有路径相对仓库根目录，先验核对SHA再解析数据。检查点仅核对哈希，不pickle载入、不执行模型。

原始CSV沿用现有评测字段，至少需要 `method,split,kappa,id,t60,scenario,moving,n_interf,snr_set,sir_set`，四项输入指标 `si_sdr_dp_in,si_sdr_in,snr_dp_in,snr_in`，主指标 `d_si_sdr_dp` 和对应 `si_sdr_dp_out`，以及计划中的次指标。按固定600 ID（MIRD连接固定元数据）检查完整性；8麦/G2仿真两个T60组各300。原始指标非有限、ID缺失/多出、场景错配不会通过“取有限交集”修复。发生冲突的重复来源阻断模型；完全相同的重复保留第一显式来源并记录，不增加n。

主账本差值使用原CSV全精度，验证 `delta=out−in`，不对分数取整。输入指标只作数值代理审查：**固定绝对容差1e−5 dB**，与科研判读的0.5/1 dB门槛完全不同。首次空跑发现旧A/B输入分数有约1e−6 dB量级差别，使用1e−10作逐位审查过严；本实现于任何新结果可见前固定这一容差，并逐次输出 `pair_audit.json` 的真实最大偏差。场景设置保持严格匹配。该代理审查不证明原波形逐字节相同；人工 `input_protocol` 资格仍须有数据版本、输入/参考哈希或相应日志证据。没有据此改动任何终点、科学门槛、检验族或事前计划。

## 资格记录：有数字也可能不可判定

每个启用模型的 `qualifications` 需要这些键：`training_protocol,checkpoint_selection,input_protocol,operator_checks,geometry`；GBmm/GB另需 `resume_budget`。每项格式：

```json
{"status":"unknown","reason":"待复核","evidence":[]}
```

复核为pass时，`evidence`必须列出真实 `[{"path":"...","sha256":"..."}]`；脚本核对证据完整性/哈希，**不会声称仅凭CSV或人工勾选已独立证明训练事实**。`fail`或`unknown`阻止正式判读，仍可保留完整配对的描述性差值。当前旧模型的训练资格没有被本轮替作者自动签字，空跑输出标SMOKE，不计入新主张。

`comparisons`还需按比较登记：C1的 `head_match,initial_function`；C2/C3的 `parameterization_recorded`；C4的 `convex_protocol`；C5及F_D/F_TD/F_R的 `no_postprocessing`。HF还核对A全部六格的非DC/DC最大残余，共7200个逐句最大值，是否超过计划1e−3；未完整提供时不自动通过。只要已有A诊断出现超阈值残余，即使其余格缺失，也阻断涉及A的正式比较，不能由人工pass覆盖数值问题。输出头映射、初始功能、恢复预算等证据应在结果可见前记录。单凭结果好不能补一个pass。

## D、冻结TD和实际WNG：两种输入方式

现有逐样本CSV保存 `target_direct_sdr_out`，但不能从它或SI-SDR倒推D。未来评测需保存**实际混合输入所选c**及其冻结目标路径。不能把网络重新喂入目标单独输入后得到的另一组c当作TD。

### 推荐：评测时导出逐句诊断CSV

调用 `diagnostics.frozen_diagnostics(...)`，输入 `H[P,F,M]`（实际运行字典）、`c[P,F,T]`、`d_nom[F,M]`、`z_direct[P,F,T]`、实际 `S_direct_actual[F,T]`、实际冻结目标时域输出 `y_direct_frozen[N]` 与参考 `x_dp[N]`，并指定 `coefficient_origin="mixture"`。可加 `Y_direct[M,F,T]`做独立有效滤波器重建。

函数计算 `h=Σconj(c_p)h_p`、非DC逐TF的 `|hᴴd−1|`的P95/RMS/最大值、DC最大值、目标增益幅相摘要、系数和残余、一般式WNG及未定义/近零事件；TD沿用仓库非尺度优化的时域误差比和EPS。检查 `Σc_p Z_direct,p` 与实际冻结目标谱一致；加Y时另检查 `hᴴY_direct`，防止共轭或dtype丢虚部的错误。它不调用网络，且仅凭数组无法证明c的估计输入；这仍需外部来源/算子资格证据。

导出返回字典并附 `method,split,kappa,id`。必须保留 `diagnostic_version,coefficient_origin,runtime_dictionary,target_output_kind,target_spectrum_reconstruction_relative_error` 和全部D列；每模型/ID唯一。清单模型项新增：

```json
"diagnostics": [{"dataset":"8mic","path":"research/my_eval/diagnostics.csv","sha256":"填写真实SHA256"}]
```

诊断TD须与原评分CSV一致（固定数值审查容差1e−3 dB）；D最大/分位必须自洽，谱重建相对误差不超过1e−5。保留返回的增益幅相、系数和残余、TF总数与WNG事件计数；有限WNG摘要缺失与全未定义WNG须分别表示，不能用0代替。上述是算子审查容差，不是新的科学优越性阈值。没有来源标记的裸D列不会被用于HF。

### 可选：一条主命令从已保存NPZ计算诊断

若已保存完整数组，可用 `diagnostic_archives:[{path,sha256}]` 指向一个显式JSON索引。索引是列表，每项 `{dataset,split,kappa,id,path,sha256}`。每个NPZ用非pickle数组保存上述H/c/d/z/S/y/x，另存字符串 `coefficient_origin="mixture"`、`stft_convention="512/sqrt-hann/128/center"`。支持可选Y_direct。固定字典须5波束、257频点、正确麦数。

主脚本在显式完成输入模式下逐个读取并释放归档，推导D/TD，还用仓库iSTFT核对实际时域冻结输出，不调用模型。来源、哈希与推导后的逐句量写入账本。同一ID不能同时给CSV和NPZ两份诊断。可优先使用小CSV，避免要求保留庞大的全部TF归档；两种方式都必须来自真实评测。当前默认清单没有诊断归档，也没有读取任何新模型文件。

## 全部输出

| 文件 | 内容 |
| --- | --- |
| `primary_56.csv` | H8=36（30差异+6 TOST）、HF=16、HG2=4；全局Holm分母固定56；可附子族Holm，但判读只用全局p |
| `secondary_102.csv` | x_ref、STOI、PESQ各34槽，分别Holm；不升级为主终点 |
| `t60_descriptive.csv` | 五个8麦比较×六格×两T60×四指标＝240行完整描述，不新增挑选p |
| `paired_t60_bootstrap.csv` | 30主差异的条件内配对bootstrap槽；固定两T60各半权重、20,000次、seed20261110 |
| `mird_condition_summary.csv` | 三比较×两场景×七指标＝42行；主指标六个条件均值精确符号秩/Holm6，其他指标描述；固定六条件t区间 |
| `mird_six_conditions.csv` | 上述逐条件完整结果；另有带SMOKE标签的旧GL−GA检查，不能当M1/M2/M3 |
| `mird_trajectory_bootstrap.csv` | 条件内轨迹簇有放回采样，簇内保留全部语音；每次先求条件均值再六条件等权，不跨条件乱采或按抽中语音数重加权 |
| `diagnostics_by_id.csv`, `diagnostics_summary.csv` | 有明确来源的逐句D/TD/WNG/重建审查，以及六格A/F摘要、全局非DC/DC最大残余、未定义WNG/近零滤波器事件总数；缺失不补造，当前逐句表为空 |
| `f_target_descriptive.csv` | 六格完整目标相对x_dp/x_ref的失真差，共12槽；与直达目标TD分开，没有新增p值 |
| `qualifications.csv` | 逐模型与逐比较的pass/fail/unknown/未执行及证据；数据完整不等于研究资格通过 |
| `legacy_smoke.csv`, `legacy_t60_bootstrap.csv` | 仅旧A−B、GL−GA的格式/配对/已知结果检查，绝不填入C/R/F/续跑主槽 |
| `source_audit.json`, `pair_audit.json` | 实际读过的文件/哈希、重复来源、输入指标最大偏差、ID哈希及源贡献 |
| `interpretation.json`, `REPORT.md`, `run_summary.json` | 事前判读、缺失原因、软件版本、源码/配置/计划哈希与运行范围 |

“未执行”的 `p_raw` 和展示用校正p留空；仅内部 `p_for_holm=1`保留槽位。资格失败但数据齐全可有描述性p，仍不用于全局判读。模型特异非有限或缺ID不通过减少n处理。

MIRD S1每条件仅一种轨迹：输出 `not_estimable_single_trajectory_condition`，CI留空，绝不用零宽区间或回退逐语音IID。六条件统计的离散p下界与共享测量限制原样披露。未来缺某个新模型或某项诊断仍可运行，脚本会生成完整槽位和原因；完整科研判读须等全部所需输入与资格齐全。

## 检查与本次范围

```bash
python -m pytest -q research/control_analysis/test_controls.py
```

测试使用手算统计量、独立合成矩阵/正弦（无训练模型）、既有A/B/G2结果和输入故障注入。包括TOST界内/界上、缺失槽Holm、精确符号秩分辨率、条件权重、单轨迹不可估计、尺度变化下D/TD与WNG的区别、共轭/iSTFT重建、pending路径从不访问、SHA与重复冲突、NaN/缺ID不删样本等。合成测试不生成或提交任何新控制模型的科研结果。实际空跑读取的全体来源见示例输出；所有新主假设仍未执行。
