# 当前版本：从建库到全量正向 FM 训练

这是当前默认操作入口；README 后半部分的负匹配、pair loss、校准实验命令不属于这次训练。
此流程复用已切分的标签和程序边界，建立四类库，但训练只使用 VLA 正片段与匹配后纳入的
CAP 成功后缀。原始专家40一并加入训练；不调用CAP-X/API，不重新运行机器人仿真。

本机／A800是验证机器，迁移机器需重新生成带本机路径的索引。源码、依赖环境、数据和
模型权重是四部分；只clone代码不能替代其余部分。

## 1. 环境与必需数据

在代码仓库根目录，先按[ENVIRONMENT.md](../ENVIRONMENT.md)准备完整训练环境，
固定 RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d`；保留对应Git checkout，后端会核验源码。
已有RLinf训练环境时只需安装新增模块，并检查依赖：

```bash
cd /path/to/vla_recovery_self_evolving
# 换成你的训练环境，不是评测环境：
source /data/pi05-work/.venv-train/bin/activate
python -m pip install --no-deps -e ./pi05_recovery_pair
python -c 'import torch, numpy, pyarrow, PIL, omegaconf, safetensors, rlinf, openpi'
cd pi05_recovery_pair
```

本流程必须有以下数据；数据目录、meta、selection、provenance以及动作映射都要完整：

```text
/data/pi05-work/
  models/RLinf-Pi05-LIBERO-SFT/
    model.safetensors
    physical-intelligence/libero/norm_stats.json
  dataset_subsets/
    rlinf_pi05_libero_fewshot40_20261021/  # 保留episodes.json和原始稀疏episode编号
    pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/
    pi05_recovery_segmented/recovery_success20_segmented/
    pi05_recovery_segmented/recovery_failure20_segmented/
```

这里是**切分后的恢复成功20／失败20库**，不是只有完整恢复成功20条的原始训练目录。
如果新机器只有“专家40＋自主20＋完整恢复20”，还缺当前分类建库所需的切分结果，
尤其是失败20的切分库。需要迁移这两个完整segmented目录；本脚本不重新转换或切分轨迹。
模型tokenizer资源沿用既有训练环境／基座资产，离线机器需提前准备，不能仅复制权重张量。

## 2. 修改三处配置

1. `configs/data_sources.example.json`：三个数据目录、norm路径、预期库数量。
   数据相对路径以 `PAIR_DATA_ROOT` 为根。可以复制成自己的JSON文件。
   当前完整数据预期四类段数为 VLA正60、VLA负40、CAP正20、CAP负362；
   输入版本不同应先核对来源，不能用删除校验掩盖缺数据。
   专家路径在scripts/my.local.env中通过PAIR_EXPERT_DATASET设置，不写入匹配数据源JSON。
2. `configs/libero_pi05_positive_fm.json`：学习率、batch、训练epoch预算等。
3. 复制并修改环境配置，填写机器路径、GPU和运行输出目录：

```bash
cp scripts/positive_fm.env.example scripts/my.local.env
# 编辑 scripts/my.local.env 和上述JSON，再加载：
source scripts/my.local.env
nvidia-smi
```

主要训练参数：

| 配置字段 | 当前值 | 含义 |
|---|---:|---|
| `fm_batch` | 64 | 全局逻辑batch，不因GPU增加而自动放大 |
| `fm_micro_batch` | 1 | 每卡一次前向/反向的窗口数 |
| `fm_epochs` | 50 | 每轮打乱全部训练窗口，无放回完整遍历50轮 |
| `learning_rate` | 1e-5 | action expert与必要投影的学习率 |
| `fm_only_updates` / `vla_fraction` | 旧兼容字段 | 当前epoch模式不使用更新上限或来源配额 |
| `pair_batch` / `eta` | 0 / 0 | 必须保持0，关闭负分支 |
| `horizon` / `action_dim` | 10 / 32 | 动作窗口长度与模型动作维度，环境7维有效 |

每个窗口每轮恰好训练一次，完整50轮；来源按实际窗口数量参与，不固定CAP配额。
当前16324窗口、FM64：每轮256更新（尾批4），总12800更新、816200次窗口呈现。
不丢弃/复制尾批、不跨轮拼批；loss除以实际全局窗口数。
多卡共用相同全局排列再分片，尾批少于卡数时空rank参与零梯度归约。
更改batch后预算自动为`fm_epochs * ceil(训练窗口数 / fm_batch)`；max_episode_reuses不控制本轮预算。
专家作为独立expert来源可追溯，不冒充VLA，不参与数值/图像匹配，既有CAP后缀准入不变。

`PAIR_SAVE_EVERY=50`、`PAIR_KEEP_LAST=3`控制保存；`PAIR_TRAIN_MEMORY_GIB=17`
控制每卡PyTorch allocator上限，加载前每卡额外要求4GiB空闲余量。
CUDA上下文/NCCL有额外开销，allocator上限不是总显存硬隔离；共享卡要先检查余量。
16卡时也默认FM64、micro1：每卡分到约4个窗口，梯度累积后做一次全局更新；不要再包DDP。

脚本prepare会把算法JSON复制为`$PAIR_RUN/algorithm.json`，并重写来源副本引用它；后续阶段
使用这一快照。要改学习率、batch或epoch预算，**在prepare之前修改JSON**。
恢复旧训练仅允许改变microbatch等明确支持项；换数据／目标／预算应开新运行目录。
`split_fractions`保留给原库构建和既定匹配分组，**最终训练资产全部为train**，不是保留评测集。
这保证重建既定匹配结果后再合并，不因跨原分组合并而额外放宽CAP准入。

## 3. 建库、匹配和准备全量训练索引

```bash
bash scripts/run_positive_fm_pipeline.sh prepare > "$PAIR_RUN.prepare.log" 2>&1
```

确保日志父目录已存在（默认 `/data/pi05-runs/`，可先`mkdir -p /data/pi05-runs`）。
此命令只做准备：CPU建立原四类段库与窗口；在`PAIR_VISION_GPU`指定的**一张卡**上提取
冻结π0.5视觉特征；按5cm/30°/RMS0.50/图像0.007匹配CAP成功窗口到VLA成功库；
从最早匹配起点纳入同一成功程序的完整后缀；最后执行`--all-train`并逐窗审计原动作、
状态和图像引用对齐。不会开始优化器更新。

当前加入专家后得到16324个逐帧窗口，全部用于训练：专家6632、中途VLA4727、自主成功3232、CAP1733。
去重真实帧16684；窗口159910个真实动作位置＋3330个重复位置＝163240个监督位置。
其中VLA中途窗口不足10真实步舍弃，自主成功和CAP成功尾部允许重复末动作，25维padding不监督。
匹配依然state_stride=10，训练单独逐帧展开，training_start_stride=1；不重新放宽准入。
输入版本变动时以实际报告为准。旧稀疏资产保留，不能将旧稀疏训练checkpoint直接续训到新资产。
准备阶段需要完整保留原窗口目录的input_manifest.jsonl、actions以及成功准入的cap_success_suffixes.jsonl；
用它们恢复原标签/事件边界和既定CAP后缀，不根据稀疏窗口盲目向前扩展。
--legacy-sparse仅用于复现历史稀疏训练，不用于本轮。

新机器必须跑该阶段；如果本机已有完整且可审计的成功准入输出，可以跳过重建和视觉匹配，
显式准备新的全train资产：

```bash
python -m recovery_pair prepare-positive-fm \
  --admission /path/to/completed_success_admission \
  --config configs/libero_pi05_positive_fm.json --all-train \
  --expert-dataset /path/to/rlinf_pi05_libero_fewshot40_20261021 \
  --output /path/to/new_all_train_prepared
python -m recovery_pair audit --prepared /path/to/new_all_train_prepared \
  --source-norm-stats "$PAIR_NORM_STATS"
```

这个复用命令只生成prepared，不创建pipeline目录结构；后续请使用README的直接train命令，
将`--prepared`替换为该新路径，而不是直接调用依赖`$PAIR_RUN/algorithm.json`的阶段脚本。
**不要复制A800的绝对路径索引到H100直接用。**

## 4. 显式启动训练

单机单卡，环境文件配置`CUDA_VISIBLE_DEVICES=可用卡号`、`NPROC_PER_NODE=1`：

```bash
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

单机16卡，先确认全部可用，再执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

这使用同一训练环境的`python -m torch.distributed.run --standalone`启动16进程，
只支持这里列出的**单机多卡**命令；不把它当作多节点启动方式。
若环境已加载，后台运行可以使用：

```bash
screen -dmS pi05_positive_fm bash -c \
  'exec bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1'
screen -r pi05_positive_fm
# Ctrl-A 再 D：退出查看，保持任务运行
```

正式训练前要做1步短检查时，可在一个新的PAIR_RUN先prepare，再执行：

```bash
export PAIR_MAX_UPDATES=1
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

随后取消限制并明确从同一输出的checkpoint继续（不要无resume重复启动）：

```bash
unset PAIR_MAX_UPDATES
export PAIR_RESUME="$PAIR_RUN/training/update_000001.pt"
bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1
```

训练只加载原始RLinf few-shot基座，冻结VLM，使用原生FM；没有pair分支、验证集、测试集
或验证早停。checkpoint记录增量可训练参数、优化器、RNG和sampler位置。
全train资产和旧828窗口训练资产不同，不能继续旧数据版本的checkpoint。

## 5. 导出权重和检查输出

选择实际存在的更新文件（以下12800对应16324窗口、全局batch64；改batch后按report预算选择）：

```bash
export PAIR_EXPORT_CHECKPOINT="$PAIR_RUN/training/update_012800.pt"
bash scripts/run_positive_fm_pipeline.sh export > "$PAIR_RUN/export.log" 2>&1
```

导出需要CPU内存、原始基座和磁盘；完整权重约7GB。输出不是完全独立的环境包，
部署仍需相同norm/tokenizer/模型配置。加载完整权重必须使用本版适配层／根evaluate.py，
它会保留FP32层精度，不能绕过后dtype重载修复。

```text
$PAIR_RUN/
  algorithm.json                 # 本次算法/训练参数快照
  sources.json                   # 本次数据来源快照
  banks/libraries/               # 四类原始段库及来源
  banks/windows/windows.jsonl    # 原始动作窗口索引
  matching/numeric/              # 数值候选
  matching/vision/               # 冻结双视角特征
  matching/admission/            # 匹配关系、成功后缀与正池
  prepared/report.json           # 应为split_policy=all_train_v1
  prepared/training_pool.jsonl   # 全部train的FM正窗口
  training/run.json              # validation_enabled=false，验证窗口0
  training/metrics.jsonl         # 原生FM loss/梯度，没有validation项
  training/status.json           # reason=epochs_complete，epochs_completed=50，position_in_epoch=0
  training/update_*.pt           # 默认只保留最新3个
  exported/full_weights.pt
```

`prepare`和`export`拒绝覆盖已有输出；失败后保留日志，排查原因再使用新目录。
训练恢复用`PAIR_RESUME`，不要删除原数据或他人输出来绕过检查。
正式运行保持`PAIR_MAX_UPDATES`为空；更新限制只用于短验证。
当前数据完成后应有`updates=12800`、`samples_seen=816200`；metrics逐步记录epoch和轮内位置。
本次不自动运行LIBERO；部署成功率评测属于单独阶段。

## 已验证范围

旧保留划分版本已在A800 GPU0、GPU2通过4次真实更新的连续／断点恢复对照。
此前稀疏全train版本在A800 GPU0实际调用本脚本prepare、train、export，训练限制1次更新；
从建库、冻结视觉匹配、1007窗口准备审计到完整权重导出、精确参数重载及原生推理均通过。
screen正常退出，耗时329.59秒，输出outputs/stage1_readme_e2e_20261031_v1/。
这不是H100安装验证、16卡实测、正式50-epoch长训或LIBERO任务成功率评测。
详细结果见[VALIDATION.md](VALIDATION.md)。

此前不含专家的逐帧版本生成outputs/positive_fm_dense_all_train_20261031_v2/，9692个窗口、
10052去重真实帧、96920个窗口动作监督位置；全部train，原始74段对齐最大误差0。
匹配结果未重做，CAP仍14个既定后缀。CPU回归93项通过。
逐帧版另在A800 GPU0完成三类真实原生FM梯度检查与1次FM64更新，峰值allocated10.57GiB，
冻结VLM SHA前后不变；输出outputs/dense_fm_gpu0_20261031_v1/。这次未重复导出/重载。

当前含专家版资产：outputs/positive_fm_expert40_dense_all_train_20261031_v1/，16324窗口。
已逐窗对照114个原Parquet（最大真实动作误差0），旧9692个rollout窗口的动作、mask、观测和范围精确不变。
96项CPU测试通过。GPU0的专家完整/补齐窗口原生FM梯度检查与一次真实更新通过；首批expert21/VLA27/code16，
冻结VLM不变，峰值allocated10.57GiB，输出outputs/expert40_dense_fm_gpu0_20261031_v1/。
没有启动正式长训、重新匹配、导出重载或LIBERO评测。

50轮采样版GPU0验证已通过：4次短更新，连续/恢复训练最终参数与日志完全一致，
冻结VLM不变；16324窗口对齐原Parquet误差0，峰值allocated12.54GiB。
结果：`outputs/epoch50_fm_gpu0_20261031_v1/report.json`。这不是正式50轮长训。
