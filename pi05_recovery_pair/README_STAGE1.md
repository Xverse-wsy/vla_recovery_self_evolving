# 第一阶段：正向片段 π0.5 FM 训练

**本次只做第一阶段，正式训练完整50个epoch。**使用已有正向片段训练π0.5，沿用原生FM loss。
负片段不参与优化，不使用pair、吸引或排斥loss；第二阶段以后另做。

## 这次跑什么

从**最初的RLinf few-shot π0.5**初始化，冻结VLM，训练action expert和必要投影。
正向监督包括：

- 原始专家40条完整示范，按episodes.json的原始编号逐帧读取。
- VLA完整自主成功轨迹。
- 已切分出的VLA正常推进片段，包括最终失败或被CAP接管轨迹中的正片段。
- 通过匹配筛选后，从最早匹配起点到同一CAP成功程序段结束的完整后缀。

**加入原始专家40；不训练VLA负片段或CAP失败片段。全部已纳入正窗口用于训练，
不留验证集／测试集，不启用验证早停。**

CAP准入只用公开任务指令、执行前本体状态和前视／腕部RGB。先数值粗筛，再用冻结的
π0.5视觉编码器精筛；阈值为位置5cm、旋转30°、标准化本体RMS0.50、图像距离0.007。
保留已有匹配分组以重建同一准入结果，再把所有准入正窗口转为train；不扩大准入范围。

训练起点改为逐帧滑动（training_start_stride=1）；匹配仍为state_stride=10，不重新匹配或扩大CAP准入范围。
例如第1帧观测监督动作1～10，第2帧监督2～11，依此类推。
动作窗口H=10，环境7维OSC动作填充到模型32维，沿用基座归一化资产，不额外转delta。
VLA中途正片段的每个起点必须有完整10真实步，不够则丢弃该起点；原始专家／完整自主成功／CAP成功后缀允许重复末动作
补齐，补齐部分参与FM监督。25个模型填充维度不参与loss。

## 1. 先准备环境、权重和数据

在新Linux GPU机器上，先按仓库根目录[ENVIRONMENT.md](../ENVIRONMENT.md)安装**训练环境**，
固定RLinf提交`c70606f08cdca259b8dec03d4430926b5b8fac9d`并保留其Git checkout。
当前模块不是完整环境安装包。已有环境时：

```bash
cd /path/to/vla_recovery_self_evolving
source /path/to/train-env/bin/activate
python -m pip install --no-deps -e ./pi05_recovery_pair
python -c 'import torch, numpy, pyarrow, PIL, omegaconf, safetensors, rlinf, openpi'
cd pi05_recovery_pair
```

本机必须有原始few-shot模型、norm/tokenizer资产，以及四个完整数据目录：

```text
$PAIR_DATA_ROOT/
  models/RLinf-Pi05-LIBERO-SFT/
    model.safetensors
    physical-intelligence/libero/norm_stats.json
  dataset_subsets/
    rlinf_pi05_libero_fewshot40_20261021/  # 保留episodes.json和原始稀疏episode编号
    pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/
    pi05_recovery_segmented/recovery_success20_segmented/
    pi05_recovery_segmented/recovery_failure20_segmented/
```

这里需要**切分后的**恢复成功／失败库，并保留meta、selection、provenance和动作映射。
失败库用于复用四类分类和提取其中VLA正片段，其负段不用于本阶段训练。
只有“专家40＋自主20＋完整恢复20”不能替代这套输入。
跨机器重新建索引，不直接使用含A800绝对路径的索引。

## 2. 配置路径、batch和预算

```bash
cp scripts/positive_fm.env.example scripts/my.local.env
# 编辑下面三处配置，然后加载机器配置：
source scripts/my.local.env
nvidia-smi
```

| 文件 | 要改什么 |
|---|---|
| `scripts/my.local.env` | 本机数据根目录、原始模型/norm路径、PAIR_EXPERT_DATASET、新输出目录、GPU号和进程数 |
| `configs/data_sources.example.json` | 三个数据目录及预期来源数量；相对路径以PAIR_DATA_ROOT为根 |
| `configs/libero_pi05_positive_fm.json` | 学习率、全局batch、microbatch、epoch预算 |

当前参数：lr=1e-5、FM全局batch64、每卡microbatch1、梯度裁剪1.0，
`pair_batch=0`、`eta=0`、`fm_epochs=50`。每个epoch独立打乱全部正窗口，无放回完整遍历，
每个窗口每轮恰好使用一次。专家保留controller=expert，始终不参与匹配。
原生FM随机采样时间与噪声，监督流速度。

**正式训练完整50个epoch，不是100次随机更新。** 当前16324个窗口、全局batch64，
每轮255个完整batch＋最后4个窗口，共256次更新；50轮共12800次更新、816200次窗口呈现。
尾批不丢弃、不复制、不跨epoch拼批，loss按实际全局窗口数归一化。
各来源权重由窗口数量决定，每批组成变化；不再固定CAP16配额。
`vla_fraction`、`fm_only_updates`、`max_episode_reuses`仅兼容旧模式，当前epoch模式不使用。
改变batch时更新数自动变为`50 * ceil(训练窗口数 / fm_batch)`，完整50遍的定义不变。
每50步保存，默认保留最新3份，可通过`PAIR_SAVE_EVERY`和`PAIR_KEEP_LAST`调整。

prepare会冻结一份参数快照到`$PAIR_RUN/algorithm.json`，后续训练使用快照。
因此先改配置，再prepare；不要训练中改数据／预算后直接续跑。
当前默认每卡allocator上限17GiB，另要求4GiB空闲余量；通过`PAIR_TRAIN_MEMORY_GIB`调整。
先检查GPU，脚本不kill或迁移其他进程。CUDA/NCCL额外开销不包含在allocator上限内。

## 3. 构建库、筛选正片段、准备全train索引

```bash
mkdir -p "$(dirname "$PAIR_RUN")"
bash scripts/run_positive_fm_pipeline.sh prepare > "$PAIR_RUN.prepare.log" 2>&1
```

该阶段依次做：已有分类建库→构建窗口→数值候选→指定单卡提取冻结视觉特征→
CAP成功后缀准入→源感知FM尾部处理＋专家40逐帧导入→`--all-train`→原数据对齐审计。
**它不启动训练，也不做负向配对。**

当前加入专家后逐帧生成16324个正窗口，全部train：专家6632、VLA中途4727、自主成功3232、CAP1733。
去重真实动作帧16684；窗口中159910个真实动作监督位置＋3330个尾部重复位置＝163240。
CAP仍是既定14个成功程序后缀；VLA中途片段末尾9个不足完整H的起点均不生成，
不会用末动作补齐。匹配与训练起点密度分开，不能将state_stride改为1来替代此流程。检查`$PAIR_RUN/prepared/report.json`：
`split_policy=all_train_v1`、`validation_enabled=false`。原始数据不修改。

## 4. 启动第一阶段训练

单卡：在环境配置中选择空闲物理卡，如`CUDA_VISIBLE_DEVICES=2`、`NPROC_PER_NODE=1`。

```bash
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

单机16卡时，先确认这些卡都可用：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

全局batch仍是64，不自动乘以16；每卡分片及梯度累积由训练器完成，不额外套DDP。
上面是单机多卡命令，不是多节点命令。希望后台运行时，用下列命令替代前台train命令：

```bash
screen -dmS pi05_stage1 bash -c \
  'exec bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1'
screen -r pi05_stage1
# Ctrl-A，再按D：离开查看但保持运行
```

训练输出位于`$PAIR_RUN/training/`：`metrics.jsonl`记录FM loss/梯度，无validation项；
`run.json`应显示`validation_enabled=false`；`update_*.pt`保存断点。
正常预算结束的`status.json`必须同时显示`reason=epochs_complete`、
`epochs_completed=50`、`position_in_epoch=0`、`samples_seen=816200`（当前数据量）。
`metrics.jsonl`逐次记录epoch、batch_in_epoch和samples_seen，方便检查真实遍历进度。
正式运行保持`PAIR_MAX_UPDATES`为空；只有短验证才设置更新限制。

需先做1步短训时，在新运行目录prepare后设置`export PAIR_MAX_UPDATES=1`，再执行train。
随后继续同一运行必须显式恢复：

```bash
unset PAIR_MAX_UPDATES
export PAIR_RESUME="$PAIR_RUN/training/update_000001.pt"
bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1
```

不能把旧不含专家资产、稀疏资产或保留划分资产的checkpoint直接resume到当前逐帧全train资产；恢复会校验数据、基座与目标版本。

## 5. 导出第一阶段模型

选择实际存在的checkpoint；以下对应16324窗口、全局batch64的完整50轮：

```bash
export PAIR_EXPORT_CHECKPOINT="$PAIR_RUN/training/update_012800.pt"
bash scripts/run_positive_fm_pipeline.sh export > "$PAIR_RUN/export.log" 2>&1
```

结果为`$PAIR_RUN/exported/full_weights.pt`，需要原始基座、CPU内存和约7GB额外权重磁盘。
部署还要保留同一norm/tokenizer与模型配置，并使用本版加载修复，避免FP32层被中间BF16转换舍入。
第一阶段到此结束；不会自动接第二阶段或运行LIBERO评测。

## 当前已完成与未执行的部分

- 当前含专家A800资产：`outputs/positive_fm_expert40_dense_all_train_20261031_v1/`，16324窗口，全部train。
- 此前不含专家逐帧资产：`outputs/positive_fm_dense_all_train_20261031_v2/`，9692窗口。
  旧版按74个原始控制段逐条核对原Parquet，观测、动作、归一化和边界对齐，真实动作最大误差0。
- CPU回归96项通过，覆盖逐帧起点、短段舍弃、VLA不补齐、CAP后缀锚点和局部标签／事件边界。
- 此前稀疏1007窗口版本的prepare→train(1更新)→export→精确重载／原生推理已通过，
  见历史`outputs/stage1_readme_e2e_20261031_v1/`；不能将其写成逐帧版的全流程新实测。
- 此前不含专家逐帧版A800 GPU0短验证通过：三类原生FM/梯度检查＋1次FM64更新，冻结VLM不变，
  峰值allocated10.57GiB；输出`outputs/dense_fm_gpu0_20261031_v1/`。未重新执行导出重载。
- 当前含专家版GPU0验证通过：专家完整/补齐窗口的原生FM梯度检查＋1次混合FM64更新。
  首批专家21、VLA27、CAP16，冻结VLM不变，峰值allocated10.57GiB。
  原有9692个rollout窗口逐项比对完全不变；输出`outputs/expert40_dense_fm_gpu0_20261031_v1/`。
- 正式50-epoch长训、H100单机16卡及LIBERO成功率评测尚未执行。

更详细的目录说明和故障处理见[RUN_POSITIVE_FM.md](RUN_POSITIVE_FM.md)，历史验证记录见
[VALIDATION.md](VALIDATION.md)。本README是本次第一阶段的首要入口。


完整50轮版本的CPU验收：109项测试通过；A800已逐轮验证真实16324窗口的采样计划，
每轮无重复无遗漏。审计输出为`outputs/epoch50_plan_audit_20261031_v1.json`。
新训练资产为`outputs/positive_fm_expert40_epoch50_20261031_v1/`；旧100更新资产保留作历史记录。
该采样审计没有执行12800次模型更新，正式长训需在目标训练机启动。

50轮采样版GPU0验证已通过：4次短更新，连续/恢复训练最终参数与日志完全一致，
冻结VLM不变；16324窗口对齐原Parquet误差0，峰值allocated12.54GiB。
结果：`outputs/epoch50_fm_gpu0_20261031_v1/report.json`。这不是正式50轮长训。
