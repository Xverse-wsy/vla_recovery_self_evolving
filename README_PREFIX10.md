# A60＋10条短VLA前缀：训练与迁移入口

> 历史10段实验留作溯源。用户当前只保留4段，且不补回任何删帧；
> 当前入口是[README_PREFIX4.md](README_PREFIX4.md)，实验`a60_prefix4`。

## 本阶段要验证什么

在原A60的专家40条＋自主成功20条上，增加自主成功20未覆盖任务的少量VLA推进前缀，
检验能否增加有效监督，同时限制前缀数据对任务分布的影响。
**本实验只加入选定的10条前缀，不加入原22个子片段、CAP动作或其他长前缀。**

实验名为`a60_prefix10`。从与A60相同的原始few-shot π0.5 checkpoint初始化，
全参数训练，包括VLM/视觉/action expert；标准原生FM loss，完整50个epoch。
不使用pair/负loss，不建立验证集或测试集，不早停。

## 数据、裁剪与密集窗口

| 来源 | 条目 | 去重真实动作帧 | 密集H10训练窗口 |
|---|---:|---:|---:|
| 原始专家 | 40 | 6632 | 6632 |
| 原自主成功 | 20 | 3232 | 3232 |
| 新短VLA前缀 | 10 | 845 | 755 |
| 合计 | **70** | **10709** | **10619** |

任务编号从0开始。前缀任务为Spatial 01/06/08、Object 09、Goal 04/05/09、
LIBERO-10 00/08/09。每任务一条，与自主成功20覆盖的18个任务不重合；
全部任务在专家40中都有完整示范，总训练任务仍为40个。

| 任务 | 裁后真实动作帧 | H10窗口 |
|---|---:|---:|
| Spatial/01 | 61 | 52 |
| Spatial/06 | 110 | 101 |
| Spatial/08 | 50 | 41 |
| Object/09 | 133 | 124 |
| Goal/04 | 33 | 24 |
| Goal/05 | 36 | 27 |
| Goal/09 | 176 | 167 |
| LIBERO-10/00 | 64 | 55 |
| LIBERO-10/08 | 105 | 96 |
| LIBERO-10/09 | 77 | 68 |

每条前缀相对最初推进片段**累计实际删除末15个动作及对应的执行前观测**。
历史HF源segmented数据已删末10步，构建工具只再删5行。
**本次上传的10个Parquet已经裁完15步，下载后和读取时都不能再裁。**

保留n行后，密集输入起点为`0,1,...,n-10`（包括n-10），每个起点监督后续10个真实动作。
每条共`n-9`窗口；不足10步的起点不参与训练，不补齐，不重复最后动作，不跨片段。
例如最初100步：裁到85步，训练起点0..75，共76个窗口，最后窗口监督75..84。
这不是再次删除剩余数据的末10步：末尾动作仍可作为完整窗口内的真实目标。

专家40和自主成功20保留A60原规则：每帧作为输入起点，尾部不足H10时允许重复本episode末动作。
新前缀禁止这样补齐。完整父轨迹instruction保留，子任务名和未来信息不作为模型输入。
RGB/state8/action7来自固定版本已转换Parquet，原值保持；不重新转动作、不翻转图像、不重采样。
数据timestamp元信息为10fps、原控制和回放频率为20Hz，分别保留。

## 训练配方与A60对齐

`config/a60_prefix10.yaml`直接继承`a60.yaml`。训练启动前逐字段检查，
只允许数据来源与实验名称不同；原A60/B80、22段实验和Stage1/Stage2保持独立。

| 项目 | 设置 |
|---|---|
| 可训练参数 | 全参数，VLM不冻结 |
| loss | 原生SFT/FM，H10×32全维均值；环境动作7维按原模型填充 |
| time/noise | 原生Beta(1.5,1)，`.999*t+.001`，高斯噪声 |
| norm | 原few-shot资产，不重新拟合、不重复归一化 |
| AdamW | lr5e-6，β=(.9,.95)，eps1e-8，weight_decay1e-10 |
| 调度/裁剪 | warmup100，原生cosine到5e-7，全局范数1 |
| 默认batch | 同A60：global100、每卡micro2，可由CLI显式调整 |
| epoch/采样 | 全部窗口train，打乱后无放回完整遍历50轮，无来源额外配额 |
| 尾批 | 不丢、不重复真实样本、不跨epoch；按实际样本数聚合 |

global100：107更新/轮、尾批19，50轮5350更新。
global128：83更新/轮、尾批123，50轮4150更新。
若历史A60运行用了128，本次也显式传128；不要把YAML默认100当作历史run的实际值。
总数据量变化会改变更新总数，调度算法和参数不变。

训练每窗口采一个flow time做FM，不运行完整多步推理去噪。
模型配置推理K=5与H=10不同；历史评测默认K=3。后续比较模型时需要显式统一评测K。

## 已有A60训练机：下载与路径

复用已有训练环境、固定RLinf源码、模型、norm、tokenizer和原40＋20数据，不重装共享环境，
不覆盖旧数据或checkpoint。新机器从零安装见[ENVIRONMENT.md](ENVIRONMENT.md)，
固定RLinf提交仍为`c70606f08cdca259b8dec03d4430926b5b8fac9d`。

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
export PI05_PREFIX10_DIR="$PI05_ROOT/dataset_subsets/pi05_prefix10_tail15_20261111/segmented"
export HF_PREFIX10_REV=ffb6d8490ee5c2711d9b5360eefc0c97ef04e82a

hf download wsy-w1k/rollout --repo-type dataset --revision "$HF_PREFIX10_REV" \
  --include 'pi05_prefix10_tail15_20261111/segmented/**' 'pi05_prefix10_tail15_20261111/assets/**' \
  --local-dir "$PI05_ROOT/dataset_subsets"
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
```

HF目录：[pi05_prefix10_tail15_20261111](https://huggingface.co/datasets/wsy-w1k/rollout/tree/main/pi05_prefix10_tail15_20261111)。
只需要新增10条已裁好的前缀，不需要旧22段、scheme1全量父轨迹、CAP匹配库或旧prepared索引。
`segmented/windows.jsonl`和原始完整instruction必须随Parquet一起下载，不能换成普通尾部补齐的取样器。
资产中没有旧机器绝对路径作为训练依赖；所有Parquet路径相对`segmented/`。

## CPU验收与原生变换

以下命令不会启动训练：

```bash
"$PI05_TRAIN_PYTHON" -m prefix_training.plan --output "$PI05_ROOT/reports/prefix10_cpu"
"$PI05_TRAIN_PYTHON" -m prefix_training.plan --world-size 16 \
  --global-batch-size 128 --micro-batch-size 2 \
  --output "$PI05_ROOT/reports/prefix10_cpu_16rank"

CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$PI05_TRAIN_PYTHON" subtask_training/native_probe.py \
  --samples "$PI05_PREFIX10_DIR/native_probe_samples.npz" \
  --checkpoint "$PI05_BASE_CHECKPOINT" --norm-stats "$PI05_NORM_STATS" \
  --reference-dataset "$PI05_AUTHOR40_DIR"
```

验收应为：70条、40任务、10619窗口、10709去重真实动作；
source_windows为6632/3232/755；前缀845行、padding0、累计trim15、50轮完整覆盖。
原生探针检查每条前缀的首/末完整窗口，共20样本，核对图像、token、state和动作归一化。
数量不符先检查路径、`episodes.json`及数据版本，不删校验绕过。

## GPU短测试：通过后再正式训练

先运行`nvidia-smi`，只选择实际分配且显存足够的GPU，不停止其他人的进程。
以下34GiB allocator＋8GiB headroom的配置要求至少43GiB初始空闲显存。

```bash
nvidia-smi
export PI05_GPUS=0  # 改成实际分配且空闲显存足够的卡
bash run_train.sh --experiment a60_prefix10 --smoke \
  --output "$PI05_ROOT/training_runs/prefix10_smoke_new" \
  --micro-batch-size 1 --gpu-memory-limit-gib 34 --gpu-headroom-gib 8
```

短测沿用A60的15样本、2轮，覆盖专家/自主/前缀三种来源，单卡micro1为10次更新；
从第2步checkpoint恢复后重放并检查权重、优化器和输入计数。
短测会缩小batch/warmup并临时保留恢复所需的checkpoint，不代表正式50轮配方。
查看`status.json`和`roundtrip_report.json`完成状态；显存不足时不启动模型训练。

本次实际验收范围见[prefix_training/validation_20261111.json](prefix_training/validation_20261111.json)。
本地和A800均通过38项测试；755个前缀窗口、1690张原始RGB及源列精确比较通过，
A800的20个原生预处理探针通过，测试核心代码与待推送版本逐文件SHA256一致。
本次GPU0短测在加载模型前被显存保护拦下：各卡仅约14～19GiB空闲，低于43GiB要求。
没有执行新方案的GPU更新，原有GPU进程均仍在运行；须在资源足够时补跑上述短测。
CPU验收、采样模拟和原生变换通过，不等于新方案GPU更新、完整50轮或LIBERO成功率已验证。

## 正式50轮与恢复

以下命令会启动长训，仅在上述验收和目标机器短测通过后显式执行。
两卡示例沿用默认global100/micro2；若实际A60用了128，追加`--global-batch-size 128`。

```bash
export PI05_GPUS=0,1
bash run_train.sh --experiment a60_prefix10 \
  --output "$PI05_ROOT/training_runs/a60_prefix10_tail15_epoch50" \
  --gpu-memory-limit-gib 40 --gpu-headroom-gib 12
```

单机16卡示例：global128/micro2，累积4次：

```bash
export PI05_GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash run_train.sh --experiment a60_prefix10 \
  --output "$PI05_ROOT/training_runs/a60_prefix10_tail15_16gpu_epoch50" \
  --global-batch-size 128 --micro-batch-size 2 \
  --gpu-memory-limit-gib 40 --gpu-headroom-gib 12
```

global必须整除GPU数×micro，代码自动计算梯度累积；micro1可降低单卡显存。
共享卡仍需allocator之外的headroom。16rank CPU采样检查不替代16卡通信/显存实测。
跨节点另需`--num-nodes`和既有`--ray-address`，不得把单机命令直接当跨节点已验证。

中断后用相同数据、GPU数、batch和输出目录，加：

```bash
--resume /absolute/path/to/author40_autonomous20_prefix10_tail15_epoch50/checkpoints/global_step_N
```

最终模型在本run的`author40_autonomous20_prefix10_tail15_epoch50/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt`。
正式过程保留最近3份checkpoint，最后一份必保存。
本实验不走原A/B总入口`run_experiment.sh all`，以免启动旧实验或长评测。

## 从固定旧HF源重新构建（迁移时通常不需要）

如需复现物理裁剪，`prefix_training/selection.json`固定这10条的任务、trial、
源Parquet、原裁剪及保留动作范围。只从固定源revision下载和校验16个Git/LFS对象，
不启动CAP、仿真或动作格式转换：

```bash
"$PI05_TRAIN_PYTHON" -m prefix_training.build \
  --source-cache "$PI05_ROOT/dataset_subsets/prefix10_source_cache" \
  --output "$PI05_ROOT/dataset_subsets/prefix10_rebuilt_new"
"$PI05_TRAIN_PYTHON" -m prefix_training.validate \
  --dataset "$PI05_ROOT/dataset_subsets/prefix10_rebuilt_new/segmented" \
  --source-cache "$PI05_ROOT/dataset_subsets/prefix10_source_cache" \
  --norm-stats "$PI05_NORM_STATS"
```

输出目录必须全新；HF已验证源cache可加`--offline`复用。
通过源RGB/state/actions/timestamp逐列比较、755窗口和校验和检查后，再将`PI05_PREFIX10_DIR`
指向新`segmented/`执行CPU验收。不同PyArrow版本可能重写成不同Parquet字节，
但必须保留全部原训练列，并在本机构建报告记录实际文件哈希。
