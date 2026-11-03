# Stage1：方案1正向片段训练、部署与运行

## 本阶段目标和当前版本

从原始 `RLinf/RLinf-Pi05-LIBERO-SFT` few-shot π0.5 初始化，用专家示范、自主成功和部署中保留的正向片段做标准 FM/SFT，完整遍历全部训练窗口50遍，导出第一阶段模型。训练后是否提高成功率，需要单独做LIBERO评测。

本页说明当前方案1，配方标识为 `stage1_a60_sft_v1`。旧16,324窗口、14段CAP后缀、冻结VLM的运行属于历史版本，不作为本页的启动依据。根目录A60/B80实验与Stage2有各自入口；本页不会自动运行它们。

接手agent按本文顺序完成：环境和下载 → 本机重建四步 → 检查建库结果 → 最多2次更新的短测试 → 检查停止状态与checkpoint。只要求部署/检查时，到短测试验收就停止；“保持50epoch”指正式预算，不代表现在启动正式训练。正式50轮另见第3节的显式续训命令。

当前资产：`dataset_subsets/pi05_recovery_scheme1_stage1_20261102/stage1_a60_sft_v1/`。
HF：[wsy-w1k/rollout/pi05_recovery_scheme1_20261102](https://huggingface.co/datasets/wsy-w1k/rollout/tree/main/pi05_recovery_scheme1_20261102)。

| 实际正向来源 | 密集训练起点/窗口 | 去重真实动作帧 |
|---|---:|---:|
| 原始专家40条 | 6,632 | 6,632 |
| VLA完整自主成功20条 | 3,232 | 3,232 |
| 新恢复成功/失败父轨迹中的VLA正常推进片段 | 3,543 | 3,903 |
| 严格匹配准入的CAP成功后缀 | **0** | **0** |
| 合计 | **13,407** | **13,767** |

当前CAP准入为0是实际严格匹配结果；不能因目标希望包含恢复行为而默认将20条CAP都加入。全部原始/转换后的轨迹及其标签仍保存，训练仅使用准入正池。失败父轨迹中的正常推进片段可以进入正池，其VLA警戒段和CAP失败动作不进入本次优化。

每轮134,070个动作监督位置中，131,370个是真实动作位置、2,700个是允许的末动作重复位置。窗口重叠，不能把这些位置数当作独立帧数。统计以本机 `prepared/report.json` 为准。

## 数据规则与建库匹配

复用既有WCM切分和程序轮次边界，先建立四类来源库，再筛选CAP、展开训练窗口：

- VLA推进片段在已有切分阶段已经物理删除最后10个动作及其输入图像/state行。训练要求后续还有完整10个真实动作，不补齐、不跨入警戒段。新机器读取现有segmented目录时不能再裁第二次。
- 专家完整示范、完整自主成功、准入CAP成功后缀：逐帧保留起点，末尾允许重复最后真实动作补足10步；重复位置参与FM。
- CAP只考虑原始成功、转换后LIBERO成功且成功代码轮次对应的那个程序段。失败尝试不因后面成功而升级为正段。
- CAP候选起点每10帧检查一次，查询VLA正库（自主成功及已有正常推进类）。专家40只在匹配后加入训练，不参与CAP匹配。
- 匹配只用同一公开任务指令、执行前state8、前视/腕部RGB：末端位置≤1cm、旋转≤5°、归一化state RMS≤0.10、每指位置差≤5mm；数值排序最多50候选，再要求冻结π0.5视觉编码器的两路图像余弦距离分别≤0.002。不读取进度评分、物体位姿、持物/接触真值或未来观测。
- 某个起点通过后，从最早匹配点纳入**同一个实际成功代码段**直到成功的完整后缀，不跨代码轮次。不能声称后续每帧都独立匹配。

匹配stride10与训练stride1不同：训练的第1帧观测监督动作1～10，第2帧观测监督动作2～11，依次滑动。若原VLA推进段长100步，先删末10步，剩90步；合法零起点为0～80，最后窗口监督80～89，不读取已丢弃的90～99。

对VLA推进前缀，任何起点到保留片段末尾不足10个真实动作，都不纳入、不补齐；裁剪后整段不足10步就没有训练窗口，恰好10步只有一个窗口。原段19步→删末10步后剩9步→0窗口；原段20步→剩10步→1窗口。代码在生成窗口和训练前审计中都检查这条规则。

保持现有7维OSC语义、state8、原norm和图像方向，不再转delta、翻图或重采样。原数据逻辑fps10与LIBERO控制20Hz分别记录，不把它们混为重采样要求。

## FM训练逻辑和参数

全参数训练：VLM、视觉编码器、action expert及投影均可训练。调用原生 `policy.compute_loss(..., train=True)`，使用原生训练图像预处理。每个窗口随机采一个flow time和高斯噪声：

\[
x_t=(1-t)A+t\epsilon,\quad
L_{\mathrm{stage1}}=L_{\mathrm{FM}}=
\operatorname{mean}_{B,H,32}\left[
\big(v_\theta(x_t,t\mid s)-(\epsilon-A)\big)^2\right].
\]

H=10；真实动作7维补到模型32维，完整32维按A60原生SFT一起求均值，包含25个模型填充维度。这与VLA时间尾部禁止补齐是不同规则。
flow time沿用原生Beta(1.5,1)及`.999*t+.001`。训练只做一个随机噪声时刻的FM前向/反向；推理才使用K=5步Euler，不在Stage1反传完整采样链。

| 参数 | 当前默认值 |
|---|---|
| 初始化/norm | 原始RLinf few-shot checkpoint及其原norm |
| 优化器 | A60共用 `config/base.yaml`，原生RLinf AdamW |
| 学习率/调度 | 峰值5e-6，warmup100更新，原生openpi_cosine降至5e-7 |
| AdamW | β=(0.9,0.95)，eps=1e-8，weight_decay=1e-10 |
| 全局梯度裁剪 | 1.0 |
| 全局batch / 每卡microbatch | 128 / 2；显存有限可将microbatch改1，保持全局128 |
| epoch | 完整50轮，每轮打乱、无放回，每个窗口恰好一次 |
| 来源比例 | 按窗口自然占比，无额外CAP/VLA配额 |
| pair/负训练、验证/测试、早停 | 均关闭 |
| 保存 | 每500更新及最后一次；保留最新3份 |

全局batch128时，每轮104个完整batch＋尾批95，共105次更新；50轮共 **5,250次更新、670,350次窗口呈现**。尾批不丢、不复制、不跨轮拼接，按实际batch窗口数归一化。多卡按相同全局排列分片、梯度SUM，不再包一层DDP。

batch可调，预算自动为 `50 * ceil(positive_windows / fm_batch)`。A60根配置global_batch_size的回退值为100，其16卡示例是16×2×4=128；本Stage1显式使用128，不能把两个默认配置字段混为同一个生效batch。

## 1. 代码、环境与资源

克隆完整适配仓库，不能只拷贝子目录。当前主线包含本配方；HF代码补丁仅用于修补此前旧clone，不需要再覆盖当前GitHub源码。

```bash
set -euo pipefail
git clone https://github.com/Xverse-wsy/vla_recovery_self_evolving.git
cd vla_recovery_self_evolving
export PACKAGE="$PWD"
export PI05_ROOT=/data/pi05-work        # 新机工作根目录
mkdir -p "$PI05_ROOT"
git rev-parse HEAD                     # 记录实际版本
```

准备Linux x86_64、Python3.12、CUDA12.6 wheel支持的NVIDIA驱动，系统依赖见 [ENVIRONMENT.md](../ENVIRONMENT.md)。只训练时不用独立评测环境、CAP-X服务或LIBERO渲染资源。安装训练环境：

```bash
# 已有RLinf时先检查版本及本地改动，不要覆盖现有目录。
git clone https://github.com/RLinf/RLinf.git "$PI05_ROOT/RLinf"
git -C "$PI05_ROOT/RLinf" checkout c70606f08cdca259b8dec03d4430926b5b8fac9d
python3.12 -m venv "$PI05_ROOT/.venv-train"
export PI05_TRAIN_PYTHON="$PI05_ROOT/.venv-train/bin/python"
PI05_PYTHON="$PI05_TRAIN_PYTHON" bash "$PACKAGE/install_environment.sh" train
"$PI05_TRAIN_PYTHON" -m pip install --no-deps -e "$PI05_ROOT/RLinf"
"$PI05_TRAIN_PYTHON" -m pip install --no-deps -e "$PACKAGE/pi05_recovery_pair"
"$PI05_TRAIN_PYTHON" -m pip check
source "$PI05_ROOT/.venv-train/bin/activate"
export JAX_PLATFORMS=cpu
```

已有完整环境可以复用，但仍须模块安装、`pip check`和固定RLinf源码校验。不能只装子目录pyproject替代完整环境。训练进程会核验RLinf源码；完整安装顺序保留fork版transformers覆盖，见安装脚本。

训练代码验证基线为 `b73a79c`，后续文档修订不改变训练算法。Stage1只需完整适配仓库和固定RLinf源码；OpenPI依赖由安装脚本安装，不需要另clone官方OpenPI、CAP-X或WCM。匹配时冻结视觉编码器，训练时模型自身的视觉编码器和VLM仍参与更新。

### 从HF下载并映射到本机路径

模型下载完整 `RLinf/RLinf-Pi05-LIBERO-SFT`，保留权重和原norm。数据仓库是 `wsy-w1k/rollout`；HF目录名与本机约定目录不同，必须按下表映射，不能直接把HF根目录当作 `dataset_subsets/`。

| HF仓库内路径 | `$PI05_ROOT/dataset_subsets/` 下的路径 |
|---|---|
| `author40/` | `rlinf_pi05_libero_fewshot40_20261021/` |
| `autonomous20/` | `pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/` |
| `pi05_recovery_scheme1_20261102/recovery_success20/` | `pi05_recovery_scheme1_stage1_20261102/recovery_success20/` |
| 同HF前缀下的 `recovery_failure20/` | 同本机前缀下的 `recovery_failure20/` |
| 同HF前缀下的 `recovery_success20_segmented/` | 同本机前缀下的 `recovery_success20_segmented/` |
| 同HF前缀下的 `recovery_failure20_segmented/` | 同本机前缀下的 `recovery_failure20_segmented/` |

以下命令用于新工作目录。HF需要认证时先在本机执行 `hf auth login`；token不写入脚本。已有数据可以按表建立软链接，跳过下载，不覆盖原目录。下载数据固定到已发布并核验的提交；不下载任何历史训练补丁或A800索引包。

```bash
"$PI05_TRAIN_PYTHON" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ['PI05_ROOT']).expanduser().resolve()
download = root / 'hf-data'
prefix = 'pi05_recovery_scheme1_20261102'
groups = ['recovery_success20', 'recovery_failure20',
          'recovery_success20_segmented', 'recovery_failure20_segmented']
mapping = {
    'rlinf_pi05_libero_fewshot40_20261021': download / 'author40',
    'pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot':
        download / 'autonomous20',
}
mapping.update({f'pi05_recovery_scheme1_stage1_20261102/{name}': download / prefix / name
                for name in groups})
# 下载前检查，避免覆盖用户现有数据；已有来源请直接复用对应目录。
for relative in mapping:
    destination = root / 'dataset_subsets' / relative
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Reuse existing data or choose a new PI05_ROOT: {destination}')

snapshot_download('RLinf/RLinf-Pi05-LIBERO-SFT',
                  local_dir=root / 'models/RLinf-Pi05-LIBERO-SFT')
snapshot_download('wsy-w1k/rollout', repo_type='dataset',
                  revision='aced94d551789d25a7a66af329f9bdcf1b232c2c',
                  local_dir=download,
                  allow_patterns=['author40/**', 'autonomous20/**'] +
                                 [f'{prefix}/{name}/**' for name in groups])
for relative, source in mapping.items():
    assert (source / 'meta/info.json').is_file(), source
    destination = root / 'dataset_subsets' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)
PY

mkdir -p "$PI05_ROOT/openpi-cache/big_vision"
# 已有同名tokenizer时保留并跳过curl，也可从A800复制该文件。
if [[ ! -s "$PI05_ROOT/openpi-cache/big_vision/paligemma_tokenizer.model" ]]; then
  curl -fL https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
    -o "$PI05_ROOT/openpi-cache/big_vision/paligemma_tokenizer.model.download"
  mv "$PI05_ROOT/openpi-cache/big_vision/paligemma_tokenizer.model.download" \
    "$PI05_ROOT/openpi-cache/big_vision/paligemma_tokenizer.model"
fi
export OPENPI_DATA_HOME="$PI05_ROOT/openpi-cache"
```

数据和模型不在GitHub。下载HF时保留以下完整目录，包括meta、selection、provenance、action mapping、segments.json及嵌入图像的Parquet：

```text
$PI05_ROOT/
  models/RLinf-Pi05-LIBERO-SFT/model.safetensors
  models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json
  openpi-cache/big_vision/paligemma_tokenizer.model
  dataset_subsets/
    rlinf_pi05_libero_fewshot40_20261021/episodes.json
    pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/
    pi05_recovery_scheme1_stage1_20261102/
      recovery_success20/
      recovery_failure20/
      recovery_success20_segmented/
      recovery_failure20_segmented/
```

专家目录除episodes.json外还需完整meta及所选Parquet；必须按原编号选40条，不能扫描全量示范。tokenizer可放默认 `~/.cache/openpi/big_vision/`，或放上图路径并 `export OPENPI_DATA_HOME="$PI05_ROOT/openpi-cache"`。norm不重新计算。

以上命令依赖同一shell中的环境变量。换终端或重新登录后，重新设置 `PACKAGE`、`PI05_ROOT`、`PI05_TRAIN_PYTHON`、`OPENPI_DATA_HOME`，再执行对应阶段；这些值都是新机绝对路径。

## 2. 新机器：从已有数据重建库

只下载四个数据目录到上图位置；本流程不需要原始混合runtime、重做OSC转换或仿真。不要把旧 `stage1_libraries.tar.gz` 中的banks/matching/prepared解压到本次重建位置；其中索引是A800绝对路径。若目录已存在，保留归档并改用新工作根目录，不能覆盖旧结果。

```bash
export TASK_PAIR="$PACKAGE/pi05_recovery_pair"
export TASK_BUILD="$PACKAGE/recovery_conversion/build_scheme1.py"
"$PI05_TRAIN_PYTHON" "$TASK_BUILD" --root "$PI05_ROOT" --pair-code "$TASK_PAIR" --stage banks
# 先nvidia-smi检查，再指定视觉编码用的一张可用卡；此处仅举例卡0。
CUDA_VISIBLE_DEVICES=0 "$PI05_TRAIN_PYTHON" "$TASK_BUILD" \
  --root "$PI05_ROOT" --pair-code "$TASK_PAIR" --stage match --device cuda:0 --memory-gib 6
"$PI05_TRAIN_PYTHON" "$TASK_BUILD" --root "$PI05_ROOT" --pair-code "$TASK_PAIR" --stage prepare
"$PI05_TRAIN_PYTHON" "$TASK_BUILD" --root "$PI05_ROOT" --pair-code "$TASK_PAIR" --stage align-training
```

四步依次为：现有四类来源建库→严格CAP匹配→逐帧全train窗口及专家40→生成A60配方资产，并逐窗确认动作、mask、起点和准入范围不变。均不更新模型。`align-training`拒绝覆盖完成目录；需要另版时用 `align_stage1_a60.py --output /new/path`。

使用本页固定数据和配置时，建库后的参考结果是13,407窗口、CAP准入0、全部train、50epoch预算。先检查本机报告；数量不同先核查来源版本和匹配报告，不为凑数量放宽门槛：

```bash
export TASK_ASSET="$PI05_ROOT/dataset_subsets/pi05_recovery_scheme1_stage1_20261102/stage1_a60_sft_v1"
"$PI05_TRAIN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['TASK_ASSET'])
c = json.loads((p / 'algorithm.json').read_text())
r = json.loads((p / 'prepared/report.json').read_text())
b = r['training_budget']
assert c['training_recipe'] == 'stage1_a60_sft_v1'
assert c['fm_epochs'] == b['target_epochs'] == 50
assert c['fm_batch'] == 128 and c['pair_batch'] == 0 and c['eta'] == 0
assert r['split_policy'] == 'all_train_v1' and not r['validation_enabled']
assert r['positive_windows'] == 13407 and r['admitted_cap_success_segments'] == 0
assert b['expected_updates'] == 5250 and b['expected_samples'] == 670350
print(json.dumps({'positive_windows': r['positive_windows'], 'training_budget': b}, indent=2))
PY
```

已有A800当前资产时跳过这四步；不能重复运行 `run_scheme1_build.sh`，它是含导出/等待转换的原始构建入口。通用 `run_positive_fm_pipeline.sh prepare` 使用历史宽松匹配，本次严格方案1不调用它。

## 3. 路径、batch、GPU、短测试与正式50轮

索引可原位复用，不合并/复制训练数据。建议把训练输出放到独立run目录，原prepared保持只读。以下在完成建库的机器执行：

```bash
export TASK_ASSET="$PI05_ROOT/dataset_subsets/pi05_recovery_scheme1_stage1_20261102/stage1_a60_sft_v1"
export PAIR_RUN="$PI05_ROOT/training_runs/stage1_scheme1_a60_v1"   # 必须是新目录
mkdir -p "$(dirname "$PAIR_RUN")"
mkdir "$PAIR_RUN"
cp "$TASK_ASSET/algorithm.json" "$PAIR_RUN/algorithm.json"
ln -s "$TASK_ASSET/prepared" "$PAIR_RUN/prepared"
export PAIR_DATA_ROOT="$PI05_ROOT"
export PAIR_PYTHON="$PI05_TRAIN_PYTHON"
export PAIR_CHECKPOINT="$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT"
export PAIR_NORM_STATS="$PAIR_CHECKPOINT/physical-intelligence/libero/norm_stats.json"
# 新机器先用micro1短测试，保持全局128；不改原prepared。
"$PI05_TRAIN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['PAIR_RUN']) / 'algorithm.json'
c = json.loads(p.read_text())
c['fm_micro_batch'] = 1
assert c['fm_batch'] == 128 and c['fm_epochs'] == 50
p.write_text(json.dumps(c, indent=2) + '\n')
PY
nvidia-smi
export CUDA_VISIBLE_DEVICES=0           # 仅例子，启动前重新选可用卡
export NPROC_PER_NODE=1
export PAIR_TRAIN_MEMORY_GIB=40          # 仅例子：本机实际空闲需额外多4GiB以上
export PAIR_SAVE_EVERY=500 PAIR_KEEP_LAST=3
export PAIR_MAX_UPDATES=2              # 只测试2次更新，不启动完整50轮
unset PAIR_RESUME
cd "$PACKAGE/pi05_recovery_pair"
bash scripts/run_positive_fm_pipeline.sh train > "$PAIR_RUN/train.log" 2>&1
```

该前台命令达到2次更新后会保存checkpoint并退出。日志重定向后终端暂时没有输出是正常的，可从另一终端 `tail -f "$PAIR_RUN/train.log"`；模型加载和保存约20.8GB checkpoint可能耗时。短测试的epoch配置仍是50，`PAIR_MAX_UPDATES=2`只限制本次执行。

短测试完成后检查状态、两次更新的有限loss/梯度，以及checkpoint可在CPU重载；不使用GPU：

```bash
"$PI05_TRAIN_PYTHON" - <<'PY'
import json, math, os
from pathlib import Path
import torch
p = Path(os.environ['PAIR_RUN']) / 'training'
s = json.loads((p / 'status.json').read_text())
assert s['reason'] == 'requested_update_limit' and s['updates'] == 2
assert s['target_epochs'] == 50 and s['samples_seen'] == 256
metrics = [json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
assert len(metrics) == 2 and all(m['fm_count'] == 128 and m['pair_count'] == 0 for m in metrics)
assert all(math.isfinite(m[k]) for m in metrics for k in ('loss', 'gradient_norm_before_clip'))
c = torch.load(p / 'update_000002.pt', map_location='cpu', mmap=True, weights_only=False)
assert c['update'] == 2 and c['identity']['objective_version'] == 'native_sft_full_action_mean_a60_v1'
assert all(k in c for k in ('optimizer', 'lr_scheduler', 'sampler', 'torch_rng', 'numpy_rng', 'python_rng'))
print('PASS: 2-update smoke stopped; checkpoint reloaded on CPU; formal budget remains 50 epochs.')
PY
nvidia-smi
```

部署/检查任务在这里停止。用户明确要求正式训练后，保留同一个run、数据和全局batch，从这2步继续到总计50epoch；不是再额外训练50轮：

```bash
unset PAIR_MAX_UPDATES
export PAIR_RESUME="$PAIR_RUN/training/update_000002.pt"
cd "$PACKAGE/pi05_recovery_pair"
screen -dmS pi05_stage1 bash -c \
  'exec bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1'
```

后台启动需要系统安装 `screen`；不用screen时可在持久终端直接运行相同的 `bash scripts/run_positive_fm_pipeline.sh train`。正式启动前重新检查选中卡的显存和进程。不要重做 `mkdir "$PAIR_RUN"`，也不要重复创建prepared软链接。

`PAIR_TRAIN_MEMORY_GIB`是每卡PyTorch allocator上限，不含全部CUDA/NCCL开销，不是资源预留或总显存硬隔离。启动前必须检查他人进程和空闲显存；不kill他人进程。本配方已有micro1实测峰值32.42GiB；17GiB冻结版预算不适用。可用显存不足时降低microbatch或换可用卡，不抬高上限挤占其他进程。

单机多卡在启动前改 `CUDA_VISIBLE_DEVICES=0,1,...` 和对应 `NPROC_PER_NODE`；仍是全局batch128。16卡micro2时每卡分到8窗口、累积4次后更新。脚本支持单机单卡/多卡；16卡跨多节点不能照抄这个standalone命令。

修改项：运行快照 `PAIR_RUN/algorithm.json` 的 `fm_batch` / `fm_micro_batch`；共享 `config/base.yaml` 的优化器和调度。模型/学习率/clip/seed/epoch等与JSON不一致会拒绝启动。数据/全局batch/目标变更用新run，不能续旧checkpoint。只改microbatch在重启时受支持，但不承诺不同卡数/微批的浮点路径逐位一致。

不能把此前验证batch2的checkpoint接到正式batch128上。只有相同数据、模型、全局batch和配置的短测试checkpoint才能按上面的命令续训。换终端后，需恢复本节中全部路径/GPU环境变量；`PAIR_RESUME`设为实际完整checkpoint，不能只恢复一个变量就启动。

## 4. 查看、恢复和导出

```bash
screen -r pi05_stage1                    # Ctrl-A再D离开查看，任务继续
cat "$PAIR_RUN/training/status.json"     # 运行期间可能还没有最终status
cat "$PAIR_RUN/training/run.json"
tail -n 3 "$PAIR_RUN/training/metrics.jsonl"
```

metrics记录epoch、轮内位置、samples_seen、学习率、FM loss与梯度。每个checkpoint保存全部可训练参数、optimizer、LR scheduler、RNG和采样游标。中断后：

```bash
unset PAIR_MAX_UPDATES
export PAIR_RESUME="$PAIR_RUN/training/update_000500.pt"  # 换成实际最新完整文件
bash scripts/run_positive_fm_pipeline.sh train >> "$PAIR_RUN/train.log" 2>&1
```

完成标准：`reason=epochs_complete`、`epochs_completed=50`、`position_in_epoch=0`、`samples_seen=670350`、`updates=5250`（全局batch128）。`reason=requested_update_limit`只表示短验证结束。

```bash
export PAIR_EXPORT_CHECKPOINT="$PAIR_RUN/training/update_005250.pt"
bash scripts/run_positive_fm_pipeline.sh export > "$PAIR_RUN/export.log" 2>&1
```

输出 `exported/full_weights.pt`，并保留norm/tokenizer与config。约7GB权重文件，训练checkpoint及优化器占用更多磁盘，启动前预留至少约100GB。评测加载完整权重必须使用本版适配层的dtype重载修复。LIBERO评测另行显式运行；本阶段不接旧Stage2入口，后者仍针对旧冻结配方。

## 验证范围

2026-11-03本机125项训练/匹配测试通过、3项因缺少RLinf/OmegaConf跳过；来源/转换/建库20项测试通过。缺少的原生环境检查在A800执行。
A800实际已完成：原生全参数FM与视觉梯度、4次GPU0有界更新（断点恢复2次/连续2次）、最终权重及全部日志精确一致。验证batch2/micro1，生产batch128；峰值allocated32.42GiB。全部13,407窗口对齐100个原Parquet，动作误差0；5个真实输入探针与A60原reader/原生变换一致，归一化动作差异<2.4e-7。
本轮另以global128/micro1、GPU0、allocator40GiB完成2次真实更新，监督256窗口，loss/梯度有限，checkpoint已保存；达到requested_update_limit后自动退出并释放GPU。输出为 `training_runs/pi05_stage1_a60_batch128_smoke_20261103_v1/`。A800本轮26项相关CPU检查通过，覆盖本机跳过项。
迁移文档补充后，11个Bash代码块和4个内嵌Python片段通过语法检查；六个HF目录的映射/重复执行保护以临时目录和模拟下载检查通过。本文建库验收和checkpoint CPU重载片段已对A800实际资产执行通过；另逐条核验3,543个VLA推进窗口均为10个真实动作、补齐步数0，相关11项窗口测试通过。本次未在空白新机重装环境或重新执行完整下载/建库，也未新增GPU更新。
详细报告见 [VALIDATION.md](VALIDATION.md) 和资产内 `gpu_validation_report.json`、`a60_data_parity.json`。它们证明检查覆盖的流程正常，不证明正式50轮完成、H100/16卡已实测或任务成功率改善。正式训练状态以独立run为准。
