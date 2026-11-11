# 当前方案：A60＋4条VLA前缀（不补回删掉的帧）

在专家40条＋原自主成功20条上，只加入Goal/09、Object/09、Spatial/01、Spatial/06。
原10前缀中的另外6段从本方案训练索引与数据文件中移除。历史10段实验保留溯源，
当前运行入口是 **`a60_prefix4`**，不是`a60_prefix10`、22段或旧方案1。

## 目标和数据范围

检验减少额外前缀范围后能否降低对A60的干扰。训练目标仍为原生正向FM/SFT，
配置直接继承A60：全参数训练，包括VLM，完整50个epoch；无pair、无负loss、无早停。
初始化、归一化、优化器、学习率及调度均与A60一致，启动时逐字段校验。

| 来源 | 条数 | 去重真实动作帧 | 密集H10窗口 |
|---|---:|---:|---:|
| 原专家40 | 40 | 6632 | 6632 |
| 原自主成功20 | 20 | 3232 | 3232 |
| 当前4前缀 | 4 | 480 | 444 |
| 合计 | **64** | **10344** | **10308** |

任务编号从0开始，总任务数仍为40。当前4个前缀任务与原自主20不重叠；专家40有相应完整示范。

| 前缀 | 保留的原动作索引（右端不含） | 帧数 | H10窗口 |
|---|---|---:|---:|
| Goal/09 trial05 | `[0,176)` | 176 | 167 |
| Object/09 trial04 | `[0,133)` | 133 | 124 |
| Spatial/01 trial08 | `[0,61)` | 61 | 52 |
| Spatial/06 trial00 | `[0,110)` | 110 | 101 |

**4段均与旧10段中各自的tail15裁剪版本逐列完全相同；不补回任何帧。**
Spatial/01完整原前缀的后段曾抓起再滑落，但当前训练仅留61行；不要把当前61行描述为
已经包含完整的抓起成功子任务。成功事件的视觉复核只用于解释选择，标签/未来观测不进入模型。
新增4段均是VLA前缀，不含CAP动作；全部使用原完整任务instruction。

当前Parquet已实际删除原推进片段末15行。下载/读取时不再裁剪。
每帧作为输入起点，只取`0..n-10`的完整真实10步窗口，共`n-9`个；不足10动作的尾部起点
不训练，不重复最后动作，不跨片段。专家40/自主20沿用A60原成功episode的尾部补齐。

移除的6段：LIBERO-10/00、08、09，Goal/04、05，Spatial/08。
它们的Parquet和窗口不在本库，读取器也拒绝误用旧10段库。

## 已有A60机器：增量下载与运行

复用原有环境、固定RLinf、模型及原40＋20数据。新机从零安装见[ENVIRONMENT.md](ENVIRONMENT.md)。
无需下载旧10段、22段、CAP父轨迹或重做动作转换。

```bash
git clone https://github.com/Xverse-wsy/vla_recovery_self_evolving.git
cd vla_recovery_self_evolving
export PACKAGE="$PWD"
export PI05_ROOT=/data/pi05-work
export PI05_TRAIN_PYTHON=/path/to/existing/train-env/bin/python
export PI05_AUTHOR40_DIR="$PI05_ROOT/dataset_subsets/rlinf_pi05_libero_fewshot40_20261021"
export PI05_AUTONOMOUS20_DIR="$PI05_ROOT/dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot"
export PI05_BASE_CHECKPOINT="$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT"
export PI05_NORM_STATS="$PI05_BASE_CHECKPOINT/physical-intelligence/libero/norm_stats.json"
export PI05_PREFIX4_DIR="$PI05_ROOT/dataset_subsets/pi05_prefix4_tail15_20261111/segmented"
export HF_PREFIX4_REV=1d3e54a94fc378a70620264a020c38d4c9c1fef0

hf download wsy-w1k/rollout --repo-type dataset --revision "$HF_PREFIX4_REV" \
  --include 'pi05_prefix4_tail15_20261111/segmented/**' 'pi05_prefix4_tail15_20261111/assets/**' \
  --local-dir "$PI05_ROOT/dataset_subsets"
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu

# 以下为CPU验收，不启动训练。
"$PI05_TRAIN_PYTHON" -m prefix4_training.plan --output "$PI05_ROOT/reports/prefix4_cpu"
"$PI05_TRAIN_PYTHON" -m prefix4_training.plan --world-size 16 \
  --global-batch-size 128 --micro-batch-size 2 --output "$PI05_ROOT/reports/prefix4_cpu_16rank"
```

必须验收为：64条/40任务/10308窗口，来源窗口6632＋3232＋444，前缀480行、padding0。
原生预处理可用`subtask_training/native_probe.py`，输入改为
`$PI05_PREFIX4_DIR/native_probe_samples.npz`（4段首/末窗口，共8个样本）。
该探针需要原训练依赖及模型资产；命令见[README_PREFIX10.md](README_PREFIX10.md)的原生变换段。

先检查可用GPU与显存；下面的短测要求至少43GiB初始空闲，不停止其他进程：

```bash
nvidia-smi
export PI05_GPUS=0  # 改成实际分配且显存足够的卡
bash run_train.sh --experiment a60_prefix4 --smoke \
  --output "$PI05_ROOT/training_runs/prefix4_smoke_new" \
  --micro-batch-size 1 --gpu-memory-limit-gib 34 --gpu-headroom-gib 8
```

仅在验收及目标机短测通过后，显式启动正式50轮：

```bash
export PI05_GPUS=0,1
bash run_train.sh --experiment a60_prefix4 \
  --output "$PI05_ROOT/training_runs/prefix4_epoch50" \
  --gpu-memory-limit-gib 40 --gpu-headroom-gib 12
```

默认global100/micro2，与A60 YAML一致；若实际A60用了128，显式追加`--global-batch-size 128`。
global100每轮104更新、尾批8，50轮5200更新；global128每轮81更新、尾批68，共4050更新。
单机16卡示例改`PI05_GPUS=0,1,...,15`，显式global128/micro2。全量逐帧无放回遍历，
无额外来源配额。推理配置K=5，FM每窗口采一个flow time，H=10；比较评测时须统一K。
恢复用相同实验、数据、GPU数/batch/输出目录，追加`--resume /absolute/path/to/checkpoints/global_step_N`。

## 从固定源重建与检查

```bash
"$PI05_TRAIN_PYTHON" -m prefix4_training.build \
  --source-cache "$PI05_ROOT/dataset_subsets/prefix4_sources" \
  --output "$PI05_ROOT/dataset_subsets/prefix4_rebuilt_new"
"$PI05_TRAIN_PYTHON" -m prefix4_training.validate \
  --dataset "$PI05_ROOT/dataset_subsets/prefix4_rebuilt_new/segmented" \
  --source-cache "$PI05_ROOT/dataset_subsets/prefix4_sources" --norm-stats "$PI05_NORM_STATS"
```

源segmented已经裁末10行，构建仅额外裁5行；可用已核验cache加`--offline`。
输出必须为新目录，禁止从已裁15行数据再次裁。构建及读取器均锁定4段和各自行数。

本次实际验证见[prefix4_training/validation_20261111.json](prefix4_training/validation_20261111.json)。
CPU数据/采样检查不等于GPU模型更新通过；本次没有启动GPU训练或正式50轮。
