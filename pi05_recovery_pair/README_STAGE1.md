# 第一阶段：正向片段 π0.5 FM 训练与迁移入口

## 阶段目标、范围和交付物

**本阶段目标：从最初的 RLinf few-shot π0.5 出发，用专家示范、自主成功行为、
部署中正常推进的 VLA 片段以及筛选后的 CAP 成功恢复片段做正向 SFT，完整训练 50 个 epoch，
产出一个吸收这些正向行为的第一阶段模型，为后续实验提供初始化。**
本阶段训练是否改善任务成功率，要在后续独立 LIBERO 评测中确认。

初始化必须使用原始 `RLinf/RLinf-Pi05-LIBERO-SFT`，不能使用我们此前微调过的模型。
冻结 VLM，训练 action expert 和必要投影；调用模型原生 `policy.compute_loss`。
每个窗口随机采样 flow time 和噪声，学习原生 FM 的流速度，按窗口有效动作元素取均值后再对 batch 取均值。
环境动作只有 7 维参与 loss，模型另外 25 个填充维度无效。

| 正向监督来源 | 本阶段如何使用 |
|---|---|
| 原始专家 40 条完整示范 | 按 `episodes.json` 的原编号读取；只加入 FM，始终不进入匹配库 |
| VLA 完整自主成功 20 条 | 作为正向训练数据，并进入 VLA 正库 |
| 恢复成功／失败父轨迹中的 VLA 正常推进片段 | 复用已有切分与标签，作为正向训练数据 |
| CAP 成功程序片段 | 在 VLA 正库匹配；从最早匹配起点纳入同一成功程序剩余的完整后缀 |

**本阶段只训练正向 FM：负片段不进入优化，不使用 pair、吸引或排斥 loss；
全部已准入正窗口用于训练，不留验证集／测试集，不启用验证早停。**
恢复失败数据目录仍是必需输入，因为其中也有已经切好的 VLA 正常推进片段。
不重新做动作空间转换、仿真回放、程序生成或数据切分；复用现有切分数据。

运行顺序是 `prepare → train → export`，每一步单独显式执行。
交付物是本机重建的库／匹配与训练索引、完整 50 轮的 checkpoint／日志，
以及 `$PAIR_RUN/exported/full_weights.pt`。
不会自动执行第二阶段、LIBERO 评测或根目录的 A60/B80 对照实验。

## 1. 迁移时必须带什么

克隆**整个适配仓库**，不要只复制这个 README 或 `pi05_recovery_pair/`：
根目录的训练依赖清单与安装脚本也需要保留。

```bash
git clone https://github.com/Xverse-wsy/vla_recovery_self_evolving.git
cd vla_recovery_self_evolving
export PACKAGE="$PWD"
export PI05_ROOT=/data/pi05-work       # 改为新机器实际工作目录
mkdir -p "$PI05_ROOT"
git rev-parse HEAD                    # 记录使用的代码版本，需包含 fefc691 的50轮实现
```

本机必须准备原始 few-shot 模型、norm、tokenizer 和以下四个完整数据目录：

```text
$PI05_ROOT/
  RLinf/                              # 固定提交的Git源码，下一节准备
  models/RLinf-Pi05-LIBERO-SFT/
    model.safetensors
    physical-intelligence/libero/norm_stats.json
  openpi-cache/big_vision/paligemma_tokenizer.model
  dataset_subsets/
    rlinf_pi05_libero_fewshot40_20261021/
    pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/
    pi05_recovery_segmented/recovery_success20_segmented/
    pi05_recovery_segmented/recovery_failure20_segmented/
```

- 专家 40：保留 `episodes.json`、完整 `meta/` 和所选 Parquet；不能根据全量 `meta/info.json` 扫描全部示范。
- 自主 20：保留 Parquet、完整 `meta/`，特别是 `meta/provenance.jsonl`。
- 切分后的恢复成功／失败库：保留 Parquet、`meta/`、`selection.json`、provenance，
  以及其引用的 action mapping 和同目录 `segments.json`。复制完整目录，别只复制 Parquet。
- tokenizer 约 4.1 MB，A800 原位置为
  `/home/wenchaoxu/.cache/openpi/big_vision/paligemma_tokenizer.model`；它不在权重文件里。
  放入上述缓存位置后，设置 `OPENPI_DATA_HOME="$PI05_ROOT/openpi-cache"`。
  也可以保留默认 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`，不设置该变量。

只有“专家40＋自主20＋完整恢复20”不够；这里必须使用**已经切分的恢复成功和失败两个库**。
不要把 A800 的 `.venv`、`outputs/` 或含 A800 绝对路径的窗口索引直接当作新机器运行资产。
新机器从这四个原数据目录重建索引。数据和权重不包含在 GitHub 代码仓库中。

## 2. 安装本阶段训练环境

平台：Linux x86_64、Python 3.12、可支持 CUDA 12.6 wheel 的 NVIDIA 驱动。
系统依赖与完整背景见 [ENVIRONMENT.md](../ENVIRONMENT.md)，
但该文件还介绍旧 A60/B80 训练和评测流程；**本阶段只安装 train 环境，按本 README 启动。**
不执行根目录的 `run_validation.sh` 或 `run_experiment.sh all`。
此阶段不用安装独立评测环境、CAP-X 感知服务或 LIBERO 场景资源，也不需要启动 Ray 集群。
训练依赖包含 RLinf/OpenPI 实现，无需另外对官方 OpenPI 仓库执行 `uv sync`。

在上一步仓库根目录执行：

```bash
# 已有RLinf目录时不要重复clone；先确认版本及本地改动。
git clone https://github.com/RLinf/RLinf.git "$PI05_ROOT/RLinf"
git -C "$PI05_ROOT/RLinf" checkout c70606f08cdca259b8dec03d4430926b5b8fac9d

python3.12 -m venv "$PI05_ROOT/.venv-train"
export PI05_TRAIN_PYTHON="$PI05_ROOT/.venv-train/bin/python"
PI05_PYTHON="$PI05_TRAIN_PYTHON" bash "$PACKAGE/install_environment.sh" train

# 必须让实际导入的RLinf可追溯到这个本地Git源码目录。
# 单独从Git URL安装的site-packages副本可能没有可用的源码目录信息。
"$PI05_TRAIN_PYTHON" -m pip install --no-deps -e "$PI05_ROOT/RLinf"
"$PI05_TRAIN_PYTHON" -m pip install --no-deps -e "$PACKAGE/pi05_recovery_pair"
"$PI05_TRAIN_PYTHON" -m pip check

source "$PI05_ROOT/.venv-train/bin/activate"
export OPENPI_DATA_HOME="$PI05_ROOT/openpi-cache"  # 使用默认~/.cache/openpi时省略
export JAX_PLATFORMS=cpu
cd "$PACKAGE/pi05_recovery_pair"
python -c 'import torch, numpy, pyarrow, PIL, omegaconf, safetensors, rlinf, openpi'
python - <<'PY'
import rlinf
from recovery_pair.source_identity import verify_installed_rlinf
from recovery_pair.openpi_backend import RLINF_COMMIT
print(verify_installed_rlinf(rlinf, RLINF_COMMIT))
PY
```

已有完整训练环境时可以跳过创建环境和重装依赖，但仍须安装本模块并通过源码校验。
`install_environment.sh train` 会在依赖安装后恢复 `rlinf-transformer-openpi` 的文件覆盖顺序；
不能只用 `pip install -r requirements.txt` 替代整套安装步骤。
安装需要可用的 PyPI/GitHub 网络；tokenizer、权重及数据自行提前准备。

## 3. 修改路径、batch 和 GPU

以下命令均在 `$PACKAGE/pi05_recovery_pair` 中，用 Bash 执行。
新登录的 shell 应重新激活训练环境并加载配置。

```bash
cp scripts/positive_fm.env.example scripts/my.local.env
# 编辑该文件及下面两个JSON，然后加载：
source scripts/my.local.env
export PAIR_PYTHON="$PI05_TRAIN_PYTHON"   # 建议在my.local.env中填训练Python的绝对路径
nvidia-smi
```

| 修改位置 | 内容 |
|---|---|
| `scripts/my.local.env` | `PAIR_DATA_ROOT`、原始模型/norm、`PAIR_EXPERT_DATASET`、新输出目录、GPU、Python及进程数 |
| `configs/data_sources.example.json` | 两个切分库＋自主库的路径、norm路径和预期来源数量；相对路径以 `PAIR_DATA_ROOT` 为根 |
| `configs/libero_pi05_positive_fm.json` | `fm_batch` 全局batch、`fm_micro_batch` 每卡微批、学习率；`fm_epochs=50` 保持不变 |

将 `PAIR_DATA_ROOT` 指向上文 `$PI05_ROOT`；专家路径独立设置，不写入匹配数据源 JSON。
模型/norm 路径必须与数据源 JSON 指向同一份资产。
`PAIR_RUN` 必须是新的绝对输出路径，并放在输入数据目录之外。
`PAIR_VISION_GPU` 只控制 prepare 的单卡视觉提取，`CUDA_VISIBLE_DEVICES` 和
`NPROC_PER_NODE` 控制 train，两个 GPU 选择相互独立。

当前：lr=1e-5、全局batch64、每卡microbatch1、梯度裁剪1.0，`pair_batch=0`、`eta=0`。
全局 batch 不随 GPU 数自动放大；例如 16 卡时仍是全局64，每卡约4个窗口，经微批累积后更新一次。
训练器手动做梯度 SUM，不额外套 DDP。
当前脚本支持**单机单卡／多卡**；如果16张H100分布在多台机器上，不能直接照抄本节单机命令。

每卡默认 PyTorch allocator 上限17GiB，另检查4GiB空闲余量；可按实际空闲显存修改
`PAIR_TRAIN_MEMORY_GIB`。CUDA/NCCL等开销不包含在allocator上限内，先检查GPU，不kill他人进程。
每50次更新保存，默认保留最新3份，由 `PAIR_SAVE_EVERY`、`PAIR_KEEP_LAST` 调整。

## 4. 建库、匹配与数据规则

CAP 准入只用公开任务指令、执行前 state8 和前视／腕部 RGB，不用特权信息或任务进度评分。
先数值粗筛，再用冻结的 π0.5 视觉编码器精筛；门槛为位置5cm、旋转30°、
标准化本体RMS0.50、图像距离0.007。CAP成功窗口查询冻结的VLA正库，不用专家作参考。
同一个成功程序有窗口匹配时，从最早匹配窗口起点纳入该程序剩余完整后缀。
先保留原匹配分组以重建既定结果，再把准入后的正窗口全部归入 train；不因合并分组扩大准入范围。

匹配起点间隔仍为 `state_stride=10`；训练在准入范围内逐帧滑动，`training_start_stride=1`。
例如第1帧观测监督动作1～10，第2帧观测监督动作2～11。H=10与flow采样积分步数K不同。
保持基座的7维OSC语义、32维模型填充和原始norm，不额外旋转图像、转delta或重采样。

| 窗口来自哪里 | 尾部规则 |
|---|---|
| 恢复父轨迹中的 VLA 正常推进片段 | 必须有完整10个真实动作；不足则舍弃该起点，**绝不补齐** |
| 专家完整示范、完整自主成功、准入CAP成功后缀 | 保留最后一帧起点；末尾重复最后真实动作补齐到10步，重复部分参与FM |

```bash
mkdir -p "$(dirname "$PAIR_RUN")"
bash scripts/run_positive_fm_pipeline.sh prepare > "$PAIR_RUN.prepare.log" 2>&1
```

prepare 依次完成：已有分类建四类库→稀疏匹配窗口→数值候选→单卡冻结视觉特征→
成功程序后缀准入→专家40导入及密集FM窗口→全train→逐窗原数据对齐审计。
它不会执行优化器更新或负向配对。新机器必须先完成此阶段，不能跳到train读取A800索引。

当前同版本数据的预期结果（输入变化时先核对来源，不要删断言强行通过）：

| 类别 | 训练窗口数 |
|---|---:|
| 专家40 | 6,632 |
| VLA中途正常推进片段 | 4,727 |
| 完整自主成功 | 3,232 |
| CAP成功后缀（14段） | 1,733 |
| 合计 | **16,324** |

去重真实动作帧16,684；窗口中159,910个真实动作位置＋3,330个重复位置＝163,240个监督位置。
原四类段库应为VLA正60／VLA负40／CAP正20／CAP负362；专家40在匹配后单独加入。
检查 `$PAIR_RUN/prepared/report.json` 应有：`state=complete`、`split_policy=all_train_v1`、
`validation_enabled=false`、`training_start_stride=1`、`expert_episodes=40`、`negative_windows=0`、`pairs=0`。

## 5. 完整训练50轮、短验证和断点恢复

每轮独立打乱全部训练窗口，**无放回完整遍历，每个窗口每轮恰好一次，完整50轮**。
各来源按实际窗口量参与，不固定CAP配额；`vla_fraction`、`fm_only_updates`、
`max_episode_reuses`仅兼容旧模式，当前epoch模式不使用。
每轮尾批不丢弃、不复制、不跨epoch拼批，loss按实际全局窗口数归一化。
16,324窗口、全局batch64时，每轮255个完整batch＋尾批4窗口＝256次更新；
50轮共 **12,800次更新、816,200次窗口呈现**。
改变batch后更新预算自动为 `50 * ceil(训练窗口数 / fm_batch)`。

prepare 会把配置冻结到 `$PAIR_RUN/algorithm.json`，后续 train 使用该快照。
**先改配置再prepare；改数据、batch、目标或预算应使用新的运行目录，不能直接续旧checkpoint。**
当前50轮版本不能resume旧稀疏／不含专家／随机有放回采样的checkpoint。

单卡正式训练（确认本机已prepare成功，且选定GPU可用）：

```bash
unset PAIR_MAX_UPDATES PAIR_RESUME
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

单机16卡时，在上条train命令前设置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
```

需要在新机器先试1次真实更新时，用下列命令替代正式train命令。
它真实占用所选GPU；prepare不需重跑，之后可从这一步继续完整50轮：

```bash
export PAIR_MAX_UPDATES=1
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
cat "$PAIR_RUN/training/status.json"
tail -n 1 "$PAIR_RUN/training/metrics.jsonl"
```

先确认短验证成功：`updates=1`、`reason=requested_update_limit`，loss和梯度有限，
存在 `update_000001.pt`，日志无异常；再执行下面的续训命令。

```bash
unset PAIR_MAX_UPDATES
export PAIR_RESUME="$PAIR_RUN/training/update_000001.pt"
bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1
```

中断恢复同理：激活同一训练环境，加载同一个 `my.local.env`，取消更新上限，
将 `PAIR_RESUME` 指向该运行中最新完整的 `update_*.pt` 后调用train。不要无resume重复启动同一输出。
只为短验证设置更新上限；正式运行必须让 `PAIR_MAX_UPDATES` 为空。
要用screen后台运行，可用下列命令**替代**上面的前台train命令：

```bash
screen -dmS pi05_stage1 bash -c \
  'exec bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1'
screen -r pi05_stage1
# Ctrl-A，再按D：离开查看但保持运行
```

## 6. 完成标准和模型导出

`training/metrics.jsonl` 记录epoch、batch_in_epoch、samples_seen、FM loss和梯度；
没有validation项。`training/run.json`应显示 `validation_enabled=false`。
正式训练成功完成时，`training/status.json` 必须同时满足：

```text
reason = epochs_complete
epochs_completed = 50
position_in_epoch = 0
samples_seen = 50 × prepared/report.json中的positive_windows
updates = 50 × ceil(positive_windows / fm_batch)
```

当前数据/B64对应 `samples_seen=816200`、`updates=12800`。
`reason=requested_update_limit`只是短验证或人为限制结束，不算完整50轮完成。

确认正式训练完成后，选择实际存在的最终checkpoint导出；以下路径对应当前数据/B64：

```bash
export PAIR_EXPORT_CHECKPOINT="$PAIR_RUN/training/update_012800.pt"
bash scripts/run_positive_fm_pipeline.sh export > "$PAIR_RUN/export.log" 2>&1
```

结果为 `$PAIR_RUN/exported/full_weights.pt`，另附导出元信息；需原始基座、足够CPU内存及约7GB权重磁盘。
保留同一norm/tokenizer与模型配置，加载使用本版适配层／根目录evaluate.py，避免FP32层被中间BF16转换舍入。
导出是本阶段最后一步，成功率评测和第二阶段以后另做。

## 已验证范围与迁移验收

本地109项测试通过；A800已审计真实数据的全部50轮采样计划，每轮无重复无遗漏。
A800当前50轮训练资产为 `outputs/positive_fm_expert40_epoch50_20261031_v1/`。
GPU0短验证 `outputs/epoch50_fm_gpu0_20261031_v1/report.json`：4次真实更新，
连续／断点恢复最终参数和全部日志完全一致，冻结VLM不变；16,324窗口对齐114个原Parquet误差0，
峰值allocated12.54GiB。稀疏旧版曾完成短训→导出→精确重载／推理；历史详情见[VALIDATION.md](VALIDATION.md)。

**完整50轮模型训练、此版新导出重载、H100单机16卡和跨机器安装均未作为已通过的实测。**
因此可以按本文迁移和操作，但目标机器仍须依次通过：环境／源码检查、prepare全量审计、
所选GPU上的1次更新，再从短验证checkpoint继续正式训练。
单机多卡与多节点不是一回事；本文未提供已验证的跨节点启动命令。

交接给另一位agent时，要求它先读本文、[AGENTS.md](AGENTS.md)和[RUN_POSITIVE_FM.md](RUN_POSITIVE_FM.md)，
按顺序完成依赖及输入检查→prepare→1步短验证→恢复并完成50轮→export，
报告实际执行范围，不把“采样计划遍历50轮”当作“模型已经训练50轮”。
