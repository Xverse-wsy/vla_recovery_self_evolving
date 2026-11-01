# Stage2：从Stage1权重继续训练，并评测三个模型

本阶段保留原生正向FM训练，再加入相似起点下正负动作的吸引和持续排斥。
训练初始化必须是本轮Stage1的checkpoint；Stage2新建优化器、采样器和更新计数，
不恢复Stage1的优化器。此入口使用完整可微去噪采样，不是单个flow时刻估计干净动作。
实现全部在新增stage2_*文件中；Stage1及已有共享执行代码、配置、数据均不修改。

## 1. 输入与数据规则

- 原始RLinf few-shot π0.5权重、原norm/tokenizer和固定RLinf源码，沿用根目录ENVIRONMENT.md。
- 同机重建的Stage1密集全train prepared：16324个FM起点、16684个去重真实动作帧。
- 同机Stage2匹配结果：`configs/libero_pi05_stage2_numeric20.json`，
  2cm/10°/标准化state8 RMS0.20/前视腕部各0.002，144个负起点、26条父轨迹、205帧去重负动作。
- 已完成50个epoch的Stage1原生训练checkpoint `update_*.pt`。
  原始few-shot权重不能冒充Stage1产物；另一数据池的checkpoint也会被拒绝。

换机器后先按README_STAGE1.md重建Stage1资产，再按README_STAGE2_MATCHING.md重建独立匹配。
不要直接复制包含A800绝对路径的索引。原Parquet、模型和tokenizer不随代码上传。

Stage2的prepare不重新匹配、切分或扩大准入范围。它复查已保存的独立起点、数值/图像门槛及mask，
然后将FM记录、真实正参考和已匹配负窗口写到新的Stage2训练目录。
保留专家、VLA、CAP来源；专家不会进入Stage1的匹配库。

FM部分沿用Stage1尾部规则：中途VLA片段不补齐，完整专家/自主成功/CAP成功允许重复末动作。
pair部分只比较双方真实动作，排除补齐和模型25维填充，去掉共享近似动作前缀，至少3步有效交集。
同一片段的下一帧必须自己匹配成功，才能形成另一个pair。全部数据train，不另留验证/测试集。

## 2. Loss、比例和预算

对正样本自己的起点和动作，使用上游原生`compute_loss`：

\[
L_{\mathrm{FM}}=\operatorname{mean}_i d_{M_i}(v_\theta((1-t_i)A_i+t_i\epsilon_i,t_i\mid s_i),\epsilon_i-A_i).
\]

在负样本自己的真实起点，从新采样纯高斯噪声开始，沿部署配置运行完整K=3步Euler，
梯度经过全部步骤，不detach；记录的正负动作只进入loss，不进入采样初始化：

\[
L_{\mathrm{Stage2}}=L_{\mathrm{FM}}+\eta\operatorname{mean}_{p}
\left[d_{M_p}(\hat A_p,A_p^+)-0.1d_{M_p}(\hat A_p,A_p^-)\right].
\]

距离在归一化动作空间计算，拉近/排斥使用同一个mask，每个窗口先按有效元素平均。
持续排斥没有hinge、截断距离或ReLU，pair loss允许为负。初始eta0.05，前10%更新线性增大。
冻结VLM，只训练action expert和必要投影；lr1e-5，梯度裁剪全局范数1.0。

每次全局更新默认：

1. 均衡选择最多4个不同失败父episode；每个抽一个事件、一个负起点和一个正参考。
2. 普通正FM窗口抽60个：专家＋VLA约45个、CAP约15个；再加入4个正参考，共64个。
3. FM与pair分别求全局均值，再相加。code不足时由专家/VLA回填；负样本从不作为FM正目标。

同一父轨迹切出更多窗口或拥有更多参考，不会因此获得更多episode权重。
默认预算是每个有配对的失败父episode最多参与50次更新，26×50=1300次episode呈现，
pair全局batch4时共325次更新。**这是Stage2的失败episode复用预算，不是Stage1的50个epoch。**
少于4个episode时缩小pair batch；没有pair时仅FM，使用配置fm_only_updates回退预算。
无需重复正向warm-up。正式Stage2训练在本次实现任务中不自动启动。

## 3. 配置与最短命令

先安装根目录训练环境和独立评测环境，再进入`pi05_recovery_pair/`：

```bash
cp scripts/stage2.env.example scripts/stage2.local.env
# 编辑本机路径、GPU、Stage1 checkpoint；不要覆盖Stage1自己的env文件。
source scripts/stage2.local.env
nvidia-smi
bash scripts/run_stage2_pipeline.sh prepare
bash scripts/run_stage2_pipeline.sh audit

# 正式训练（只有明确要运行正式预算时执行）
bash scripts/run_stage2_pipeline.sh train

# 中断续跑：保持数据、逻辑batch、预算、初始化一致，允许调整microbatch/卡数。
export S2_RESUME="$S2_RUN/training/update_000050.pt"
bash scripts/run_stage2_pipeline.sh train

# 导出准确指定的Stage2 checkpoint到新的完整权重目录
export S2_EXPORT_CHECKPOINT="$S2_RUN/training/update_000325.pt"
bash scripts/run_stage2_pipeline.sh export
```

训练配置是独立的`configs/libero_pi05_stage2_numeric20.json`。修改该文件的
fm_batch、pair_batch、fm_micro_batch、pair_micro_batch、learning_rate、max_episode_reuses等，
然后在新输出目录prepare。配对门槛、min_pair_steps和动作表示必须保持当前选定规则。
不要改Stage1的configs/libero_pi05_positive_fm.json或共享configs/libero_pi05.json。

全局FM64、pair4不随GPU数自动乘大；每卡microbatch控制内存，训练引擎自动分摊全局样本并累积。
16卡单机可设S2_NPROC=16及对应CUDA_VISIBLE_DEVICES。多节点另设S2_NNODES、S2_NODE_RANK、
S2_MASTER_ADDR、S2_MASTER_PORT；所有节点必须有一致路径、代码、模型和prepared。
torchrun/NCCL需要在目标机器验证；本机测试不能代替16张H100实测。
默认每进程20GiB allocator上限，加载前须另有4GiB空闲；可按目标机器资源调整，不能kill其他用户进程。

## 4. GPU短验证：不等于正式训练

```bash
CUDA_VISIBLE_DEVICES=0 "$S2_PYTHON" -u -m recovery_pair.stage2_validation \
  --prepared "$S2_RUN/prepared" --config "$S2_CONFIG" \
  --stage1-checkpoint "$S2_STAGE1_CHECKPOINT" --base-checkpoint "$S2_BASE" \
  --norm-stats "$S2_NORM" --output "$S2_RUN/gpu_validation" --max-cuda-memory-gb 20
```

这个验证入口仅运行：一次更新＋重载续训第二次更新、独立连续两次更新，共4次真实更新；
要求最终训练参数哈希和逐步metrics精确一致、冻结VLM不变、pair确实参加更新，
然后导出并在GPU重载，要求训练/冻结参数精确一致、原生推理动作有限。
短验证显式允许不完整Stage1权重；正式train默认拒绝它。
GPU短验证不运行LIBERO，也不能报告任务成功率；下面独立执行小规模仿真检查。

## 5. 评测：fewshot、Stage1、Stage2

Stage1完整权重路径填入S2_STAGE1_EXPORTED。若尚未导出，可用下面新增的只读导出入口，
不会在Stage1训练目录中写临时文件，输出目录必须另建。
S2_EVAL_PYTHON指向独立评测环境，S2_EVAL_PACKAGE指向包含stage2_evaluate_parallel.py的完整适配仓库。
S2_EVAL_GPUS为当前节点可用物理GPU列表。评测每卡一个模型/环境，按episode分片，不使用训练batch。

```bash
bash scripts/run_stage2_pipeline.sh export-stage1
# 完整评测：每模型4个suite×10任务×50初始状态=2000条；不启用WCM或CAP。
bash scripts/run_stage2_pipeline.sh eval fewshot
bash scripts/run_stage2_pipeline.sh eval stage1
bash scripts/run_stage2_pipeline.sh eval stage2
bash scripts/run_stage2_pipeline.sh compare

# 小规模流程验证（替换当前输出根，或使用新的S2_RUN，避免混入正式输出）
# 以下只演示Stage2；各模型若要对照，必须使用相同参数与初态。
bash scripts/run_stage2_pipeline.sh eval stage2 \
  --suites libero_object --tasks 0 --trials 1 --max-steps 20 --no-video

# 评测续跑保持分片数/权重/推理参数不变
bash scripts/run_stage2_pipeline.sh eval stage2 --resume
```

正式评测max_steps=0使用各suite原生预算；有界smoke显式记为子集，不能作为完整成功率。
成功来自LIBERO终止判定，不以程序没报错或动作有限代替。
评测H=10、K=3，默认每次执行5个动作，与既有基线一致；有export配置时拒绝不同H/K。
评测默认要求每张卡至少29GiB空闲（16GiB allocator与共享余量）；不足时不启动该卡。
跨节点评测可额外传--num-shards和--shard-offset，各节点使用同一输出、不同全局分片，
全部分片完成才汇总；最终可用--merge-only重新检查完整性。

比较脚本要求三个模型同一任务/初态集合、norm、K、动作执行长度及预算，正式比较必须2000条齐全。
训练初始状态重合信息只从显式S2_TRAINING_INITIALS读取canonical [suite,task,trial]三元组；
未提供时记录unknown，不伪造“没有训练重合”。此信息仅用于结果标记，不进入模型输入。

## 6. 输出、边界与验收

```text
S2_RUN/
  prepared/{windows.jsonl,pairs.jsonl,fm_ids.json,config.json,checksums.json,report.json,actions/}
  training/{run.json,status.json,metrics.jsonl,update_*.pt}
  exported/{full_weights.pt,config.json,source.json}
  evaluation/{fewshot,stage1,stage2}/{parallel.json,status.json,workers/,episodes/,summary.json}
  comparison/{comparison.json,comparison.md}
```

新阶段不是--resume Stage1。Stage2续跑只接受完全相同的Stage2身份与初始化来源。
checkpoint记录原始base/norm哈希、Stage1checkpoint哈希、原Stage1数据池哈希、Stage2prepared哈希和目标版本。
Stage1原目录和权重保持只读，不在其中写临时文件；输出必须另建目录。
Stage2新文件可以调用已验证的共享FM/采样/加载函数，但不改变它们或Stage1流程。
实际验证结果及局限见VALIDATION.md；H100多机、正式325更新、完整三模型6000条评测不由短验证代替。

2026-11-01已完成121项CPU测试，以及A800 GPU0的4次真实优化更新、断点/连续训练精确一致、
导出重载和原生推理检查，峰值allocated约12.54GiB。另完成三个模型各2条、每条最多20步的
LIBERO流程验证及相同初态比较。短验证使用尚未完成50epoch的Stage1权重；正式训练仍要求
完整Stage1产物，未自动启动长训或完整评测。详细输出路径与实际验证范围见[VALIDATION.md](VALIDATION.md)。
