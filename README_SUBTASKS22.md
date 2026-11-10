# A60＋原始22个子任务片段：训练与迁移入口

## 目标、范围与状态

本实验检验：在专家40＋VLA完整自主成功20的A60数据上，增加rollout中人工复核的
连续推进子片段，能否改善标准π0.5 SFT。**只用原来的22段，新增46段不纳入。**
实验名为`a60_subtasks22`，保留原有A60/B80及方案1 Stage1。

从与A60相同的原始`RLinf/RLinf-Pi05-LIBERO-SFT`初始化，完整训练50个epoch。
`config/a60_subtasks22.yaml`直接继承A60；启动时逐字段核对数据/输出名称之外的配方。
无新匹配、负训练、pair分支或验证早停。

本机和A800的32项CPU回归测试通过，真实数据完整验收及5个样本的原生预处理探针通过。
A800的HF HTTPS仍报TLS断连，本次复用服务器原有12条转换父轨迹恢复原22段，
核对12个父文件SHA256及22段全部训练列，与HF发布版逻辑数据一致。
GPU4的10次更新及从第2步恢复的8次重放全部通过，权重/优化器/输入计数完全一致。
峰值allocated32.363GiB、reserved33.973GiB；测试退出后GPU4显存恢复到启动前水平。
正式50轮没有启动。
短测只验证执行与恢复，完整50轮训练和LIBERO成功率需后续实验验证。

## 数据和尾部规则

| 来源 | 条目 | 去重真实动作 | 密集训练起点/窗口 |
|---|---:|---:|---:|
| 原始专家 | 40条完整episode | 6632 | 6632 |
| 原有VLA自主成功 | 20条完整episode | 3232 | 3232 |
| 复核子片段 | 22段，来自12条父episode | 1311 | 1194 |
| 合计 | 60条完整episode＋22段 | **11175** | **11058** |

所有窗口为train，每轮打乱、无放回完整遍历50遍。专家和自主成功沿用原位读取，
没有复制/拼接成一个合并数据集。

- **训练instruction使用父轨迹原始完整任务指令**。子任务名、stage、milestone、
  人工评语和未来观测都不作为模型条件。
- 13个中间片段：n个动作仅保留零起点`0,1,...,n-10`，包括n-10，
  共`max(0,n-9)`窗口。每个窗口监督随后10个真实动作，不重复末动作。
  不足10步的整段无窗口；本次22段均至少10步。
- 9个最终成功片段：最后一个动作在已有转换回放中触发LIBERO成功判定，
  可按A60成功示范的规则保留末尾起点，并重复本段最后动作补足H10。
  父轨迹最终成功不能让中途片段取得此权限。
- 窗口不跨片段、controller或代码轮次。这里仅丢弃不足10个真实动作的**输入起点**，
  不额外物理删除每段最后10个动作。
- RGB/state8/action7来自原始Parquet，不从标注视频反解，不翻转、不转delta、不重采样。
  `[a,b)`为转换后父Parquet的action行号；补偿后的`source_action_step`不能代替行号。
  数据timestamp fps10与native control_hz20分别保留。

子片段窗口含11940个动作位置，其中405个是仅最终成功尾部允许的重复位置；
**中间片段重复位置为0**。重叠窗口位置数不是独立帧数。
标签保留人工连续推进复核依据，没有声称逐动作最优或新仿真验证。

## A60配方

| 项目 | 本实验直接继承的设定 |
|---|---|
| 可训练参数 | 全参数：VLM、视觉编码器、action expert及投影 |
| loss | 原生SFT/FM，原生训练图像处理，完整H10×32维均值 |
| 动作和norm | 环境7维＋模型25维填充，原始few-shot norm，不重新拟合 |
| flow time/noise | 原生Beta(1.5,1)、`.999*t+.001`、高斯噪声 |
| AdamW | lr5e-6，β=(.9,.95)，eps1e-8，weight_decay1e-10 |
| 调度/裁剪 | warmup100，原生openpi_cosine到5e-7，全局范数1 |
| 默认batch | 同A60配置：global100、每卡micro2；CLI可调 |
| epoch/比例 | 完整50轮，每窗口每轮一次，按实际数量，无来源配额 |
| 尾批/恢复 | 不丢/复制尾批、不跨epoch；复用A60的checkpoint、数据身份、游标和RNG恢复 |

数据变多后总更新数自动重算，调度算法和参数保持一致。
global100：111更新/轮、尾批58、50轮5550更新。
global128：87更新/轮、尾批50、50轮4350更新。
若历史A60实际用了128，本次也显式传128；YAML默认100不能代表某次历史运行的实际batch。

训练每个窗口只采一个flow time做FM，不运行完整多步去噪链。
模型配置的推理K=5和动作窗口H=10不同。历史A60评测脚本默认K=3，
后续评测比较时需显式统一评测K，不能用训练字段推断历史评测步数。

## 已有训练机：代码、下载和路径

目标机已完成A60/B80/旧Stage1时，复用现有训练环境、固定RLinf源码、few-shot权重、
norm、tokenizer、专家40与自主20，不覆盖旧checkpoint/数据。
从零安装见[ENVIRONMENT.md](ENVIRONMENT.md)。`subtask_training/requirements.txt`
仅供独立CPU数据工具环境，不用于替换现有训练依赖。

```bash
git clone https://github.com/Xverse-wsy/vla_recovery_self_evolving.git
cd vla_recovery_self_evolving
export PACKAGE="$PWD"
export PI05_ROOT=/data/pi05-work
export PI05_TRAIN_PYTHON=/path/to/existing/train-env/bin/python
git rev-parse HEAD

# 指向当时用于A60的原数据；专家必须保留episodes.json原选择。
export PI05_AUTHOR40_DIR="$PI05_ROOT/dataset_subsets/rlinf_pi05_libero_fewshot40_20261021"
export PI05_AUTONOMOUS20_DIR="$PI05_ROOT/dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot"
export PI05_BASE_CHECKPOINT="$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT"
export PI05_NORM_STATS="$PI05_BASE_CHECKPOINT/physical-intelligence/libero/norm_stats.json"
export PI05_SUBTASK22_DIR="$PI05_ROOT/dataset_subsets/pi05_subtasks22_20261110/segmented"

export HF_SUBTASK22_REV=d41d10e8701771c087e73d9296f62bc4c78487c3
hf download wsy-w1k/rollout --repo-type dataset --revision "$HF_SUBTASK22_REV" \
  --include 'pi05_subtasks22_20261110/segmented/**' 'pi05_subtasks22_20261110/assets/**' \
  --local-dir "$PI05_ROOT/dataset_subsets"
```

HF目录：[pi05_subtasks22_20261110](https://huggingface.co/datasets/wsy-w1k/rollout/tree/main/pi05_subtasks22_20261110)。
新增Parquet已切成22段，内含原始RGB/state/action。专家40和自主20无需重复下载。
不需要旧方案1的长前缀、CAP匹配结果或prepared索引。
新数据内路径相对`segmented/`，迁移时不用A800/Mac旧绝对路径。
**必须使用本模块窗口索引，不能换成会给所有子片段尾部重复动作的普通LeRobot取样器。**

## CPU验收：不会启动训练

在仓库根目录，加载上节环境变量：

```bash
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
"$PI05_TRAIN_PYTHON" -m subtask_training.plan --output "$PI05_ROOT/reports/subtasks22_cpu"

# 16卡取样计划的CPU验收，不会启动GPU进程。
"$PI05_TRAIN_PYTHON" -m subtask_training.plan --world-size 16 \
  --global-batch-size 128 --micro-batch-size 2 \
  --output "$PI05_ROOT/reports/subtasks22_cpu_16rank"
```

检查`preflight.json`：passed、82个episode条目、40任务、11058窗口、
source_windows=6632/3232/1194、parent_instructions_only=true、intermediate_tail_padding=0、
A60配方逐字段一致、50轮完整覆盖。数量不符先查路径和专家episodes.json，不能删校验绕过。
此命令不导入Ray、不加载模型或CUDA。

## 原生变换和GPU短测试

CPU验收通过后，先运行原生变换探针，再运行GPU短测：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$PI05_TRAIN_PYTHON" subtask_training/native_probe.py \
  --samples "$PI05_SUBTASK22_DIR/native_probe_samples.npz" \
  --checkpoint "$PI05_BASE_CHECKPOINT" --norm-stats "$PI05_NORM_STATS" \
  --reference-dataset "$PI05_AUTHOR40_DIR"

nvidia-smi
export PI05_GPUS=4  # 改成实际分配的卡；启动前重新检查空闲显存
bash run_train.sh --experiment a60_subtasks22 --smoke \
  --output "$PI05_ROOT/training_runs/subtasks22_smoke_new" \
  --micro-batch-size 1 --gpu-memory-limit-gib 34 --gpu-headroom-gib 8
```

第一条只测试CPU原生变换/EnvIO/tokenizer，不加载模型权重。
smoke沿用A60的15样本/2轮，覆盖三种来源并检查checkpoint恢复；
它会为短测试缩小batch/warmup，不是正式50轮配方。
根据实际显存调整allocator和microbatch并保留headroom。入口要求每卡初始空闲显存至少为
`gpu-memory-limit-gib + gpu-headroom-gib + 1` GiB；上例需要43GiB。
2026-11-10实际短测选GPU4，启动时约44GiB空闲；该显存状态不是后续运行的资源预留。
allocator上限约束本进程PyTorch分配，CUDA上下文等另占显存；不足时测试退出，不停止其他进程。
GPU资源检查不能代替实际短测试。成功后检查本run的`status.json`为`complete`、
`roundtrip_report.json`的原跑/恢复结果一致，并查看`progress_rank_*.json`的峰值显存。
单卡micro1短测为10次更新，并从第2步恢复后重放；短测会保留恢复所需的全部checkpoint，
修复了默认最近3份保留策略提前清理第2步的情况。正式训练仍按原规则保留最近3份。

## 正式50轮与恢复

正式训练单独显式启动；以下为单机16张H100、与A60的16卡global128配置对齐：

```bash
export PI05_GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash run_train.sh --experiment a60_subtasks22 \
  --output "$PI05_ROOT/training_runs/a60_subtasks22_epoch50" \
  --global-batch-size 128 --micro-batch-size 2 \
  --gpu-memory-limit-gib 40 --gpu-headroom-gib 12
```

global需被GPU数×micro整除；16卡micro2/global128累积4次，micro1则累积8次。
多卡直接复用A60的RLinf/Ray及归约。CPU检查了16rank取样和尾批权重，
真实16卡通信/显存/GPU数值行为仍需目标机smoke。
跨节点像A60一样另传`--num-nodes`、`--ray-address`并准备共享路径。

恢复使用本实验同数据、GPU数、batch、输出目录，加上：

```bash
--resume /absolute/path/to/author40_autonomous20_subtasks22_epoch50/checkpoints/global_step_N
```

最终权重位于本run的`.../checkpoints/global_step_N/actor/model_state_dict/full_weights.pt`。
数据身份检查会拒绝拿A60或旧Stage1的checkpoint直接恢复本实验输入游标。

## 重建及本次证据

通常下载已构建数据即可。重新构建需原22段审阅manifest、V1 sources里的数值列、
动作映射、代码和同级source_trees.json，这些来源材料不在代码仓库。
父数据固定revision=`aced94d551789d25a7a66af329f9bdcf1b232c2c`；
原22段审阅manifest SHA256=`4c221b56cd94419925b87ecd816d5d5bcf7cd33070d4085a0811a461448f68a9`。

```bash
python -m subtask_training.fetch --review /path/to/review22 \
  --numeric-sources /path/to/v1/sources --output /path/to/new_output --fetch-normalization
python -m subtask_training.build --review /path/to/review22 \
  --numeric-sources /path/to/v1/sources --output /path/to/new_output \
  --horizon 10 --tail-policy success_endpoint_repeat
python -m subtask_training.validate --output /path/to/new_output --review /path/to/review22 \
  --norm-stats /path/to/new_output/assets/norm_stats.json
```

本机CPU证据：32项测试；22段父行/图像字节核对、2622张RGB解码、1194窗口完整加载、
单/双worker读取、归一化与既有训练器误差0；真实40＋20＋22混合batch；
默认及16rank配置的50轮完整取样；原尾批参考的144种计划、403次梯度比较、
366个零权重同步batch。初版CPU报告随HF固定数据revision提供；最新A800验证范围如下。

## A800实际验证（2026-11-10）

- 原有训练环境、few-shot权重、norm、tokenizer及RLinf固定版本原位复用，未重装共享依赖。
- 本机及A800各32项CPU回归测试通过，覆盖单卡/双卡短测恢复所需checkpoint保留，以及正式最近3份策略。
- A800完整混合数据验收：82条数据条目、40任务、11058窗口；原22段1194个窗口全部核对，
  中间片段补齐为0，完整父任务instruction保持一致，50轮采样完整。
- 原生RLinf/OpenPI探针5个真实样本通过：图像和token/mask完全一致；动作归一化最大误差
  2.77e-7，state最大误差2.73e-8。该探针不加载模型权重。
- GPU4全参数SFT短测通过：3353433872个可训练参数，VLM和action expert均参与；
  15窗口、global3/micro1、2个短测epoch、10次更新，loss及梯度均有限。
- 第2步checkpoint加载后重放8次更新，最终权重、优化器状态、数据游标和计数完全一致。
  短测发现并修复早期checkpoint被保留策略提前删除的问题；32项CPU测试已覆盖此回归。
- allocator上限34GiB、额外headroom8GiB；包括恢复重放的峰值allocated32.363GiB、
  reserved33.973GiB。退出后GPU4恢复45261MiB空闲，启动前已有GPU进程均仍在。
- A800报告：`$PI05_ROOT/reports/subtasks22_a800_20261110_v2/`；
  短测输出：`$PI05_ROOT/training_runs/subtasks22_a800_smoke_20261110_v2/`。
  测试完成后清理了本次第4/6/8步冗余checkpoint，保留第2/10步及全部报告。
- 可随代码查看[验证摘要](subtask_training/validation_a800_20261110.json)。

本次只验证单张A800；H100/16卡硬件通信、正式50轮和LIBERO成功率未运行。

本次A800数据目录为
`$PI05_ROOT/dataset_subsets/pi05_subtasks22_parent_restore_20261110_v1/segmented`，
使用它时显式设置`PI05_SUBTASK22_DIR`。Parquet writer为19.0.1，发布版为26.0.1，
文件封装字节哈希可能不同；父文件哈希和输出全部训练列均逐一验证一致。
新机器仍直接下载上面的HF固定版本，无需重复恢复父轨迹。
