# 两个独立环境：训练与评测

适用 Linux x86_64、Python 3.12、NVIDIA GPU。训练使用 A800 已验证的
PyTorch 2.7.1 / CUDA 12.6 组合；评测使用已验证的 PyTorch 2.6.0 / CUDA 12.4
组合。不要把两份 requirements 装进同一个环境。

`requirements.txt` 是训练依赖快照；`requirements-eval.txt` 是评测入口，引用同目录
的 `constraints-eval.txt` 固定其间接依赖。请保留这些文件的相对位置。

## 1. 工作目录和源码

在当前代码仓库根目录执行，并把 `PI05_ROOT` 改成新机器实际工作目录：

```bash
export PACKAGE="$PWD"
export PI05_ROOT=/data/pi05-work
mkdir -p "$PI05_ROOT"
```

现有 RLinf checkout 应放在 `$PI05_ROOT/RLinf`，也可以将该位置软链接到已有仓库。
本适配器依赖源码中的 `examples/embodiment/config/env/*.yaml`，因此安装 Python
包不能代替这个源码目录。固定到已验证的提交：

```bash
git -C "$PI05_ROOT/RLinf" checkout c70606f08cdca259b8dec03d4430926b5b8fac9d
```

评测沿用 A800 的 OpenPI 源码导入方式，另外准备固定版本的 OpenPI：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$PI05_ROOT/openpi"
git -C "$PI05_ROOT/openpi" checkout 215abfb217dbac7d5f1273282331b9b1866c0479
```

如果已有此目录，检查提交即可，不要重复 clone。评测环境不安装 `rlinf-openpi`，
也不对该 OpenPI checkout 运行 `uv sync`；其依赖由评测 requirements 固定。

## 2. 安装两个环境

前提是系统已有 Python 3.12（或使用 Conda 创建对应的独立环境）。以下命令
不会启动模型训练或评测：

```bash
python3.12 -m venv "$PI05_ROOT/.venv-train"
python3.12 -m venv "$PI05_ROOT/.venv-eval"

export PI05_TRAIN_PYTHON="$PI05_ROOT/.venv-train/bin/python"
export PI05_EVAL_PYTHON="$PI05_ROOT/.venv-eval/bin/python"

PI05_PYTHON="$PI05_TRAIN_PYTHON" bash "$PACKAGE/install_environment.sh" train
PI05_PYTHON="$PI05_EVAL_PYTHON" bash "$PACKAGE/install_environment.sh" eval
```

安装训练依赖需要访问两个固定 Git 提交（RLinf 和 LeRobot）、PyPI；评测还需要
访问 PyTorch 官方 CUDA 12.4 wheel 索引。请在运行安装命令的 shell 中配置可用代理。

训练环境中的 `transformers` 和 `rlinf-transformer-openpi` 共用模块目录。
安装脚本会在所有依赖装完后，最后重新安装 OpenPI 的实现，匹配 A800 上已验证的
文件内容。只运行 `pip install -r requirements.txt` 不能保证这两个包的文件覆盖
顺序，因此建议使用上面的安装脚本。脚本最后对每个环境运行 `pip check`。

如果手动安装，对应命令是：

```bash
"$PI05_TRAIN_PYTHON" -m pip install -r "$PACKAGE/requirements.txt"
"$PI05_TRAIN_PYTHON" -m pip install --force-reinstall --no-deps rlinf-transformer-openpi==4.53.2
"$PI05_TRAIN_PYTHON" -m pip check

"$PI05_EVAL_PYTHON" -m pip install -r "$PACKAGE/requirements-eval.txt"
"$PI05_EVAL_PYTHON" -m pip check
```

## 3. 系统库和非 Python 资源

pip 不安装 NVIDIA 驱动、系统图形库、模型权重、数据和场景资源。Ubuntu/Debian
常用的系统包是 `git`、`build-essential`、`libegl1`、`libgl1`、`libglib2.0-0`、
`ffmpeg`，以及与 Python 版本匹配的开发头文件。驱动必须支持所选 CUDA wheel；
无须为了这套脚本单独启动 JAX GPU，运行时会设置 `JAX_PLATFORMS=cpu`。

将已有资源放在以下位置，或使用软链接。数据不需要重新合并、导出：

```text
$PI05_ROOT/
  RLinf/
  openpi/
  models/RLinf-Pi05-LIBERO-SFT/
    model.safetensors
    physical-intelligence/libero/norm_stats.json
  dataset_subsets/
    rlinf_pi05_libero_fewshot40_20261021/
    pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot/
    pi05_long_recovery20_20261028/recovery_success20_lerobot/
  libero-resources/
    assets/
    bddl_files/
    init_files/
```

三个数据目录都保留完整 `meta/`；原始 40 条的 `episodes.json` 和 rollout 数据的
`meta/provenance.jsonl` 也要保留。若使用别的目录名，在 `experiment.env` 设置
`PI05_AUTHOR40_DIR`、`PI05_AUTONOMOUS20_DIR`、`PI05_RECOVERY20_DIR`，或修改
`config/paths.yaml`。训练与评测来源统计会读取同一套路径，无需修改 Python 代码。

另外复制 A800 上的 tokenizer：

```text
/home/wenchaoxu/.cache/openpi/big_vision/paligemma_tokenizer.model
```

在新机器放到 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`；或放到自定义
缓存目录下的 `big_vision/`，然后设置 `OPENPI_DATA_HOME=/absolute/path/to/cache`。
该文件约 4.1 MB，不在模型权重目录中。

复制 `libero-resources` 后，使用评测环境更新本机绝对路径和 assets 链接：

```bash
"$PI05_EVAL_PYTHON" "$PACKAGE/configure_resources.py" --resources "$PI05_ROOT/libero-resources"
```

该脚本会备份已有资源配置和 venv 内已有的 assets 入口，不更改资源文件内容。

## 4. 安装后检查与短验证

```bash
"$PI05_TRAIN_PYTHON" "$PACKAGE/verify_environment.py" --mode train --resources
"$PI05_EVAL_PYTHON" "$PACKAGE/verify_environment.py" --mode eval --resources
```

检查 Python/关键包版本、源码提交、训练用 Transformers 文件、模块导入、数据
元文件、norm stats 和 tokenizer；评测还检查四个 suite 的初始化状态。不会加载
大模型、训练或执行 rollout。可加 `--cuda` 检查设备可见性和 NCCL 支持，但这不等于
完成多卡通信测试或 EGL 实际渲染测试。

确认选定 GPU 空闲后，执行之前的短训练、断点恢复和真实环境评测验证：

```bash
PI05_VALIDATION_GPUS=0,1 bash "$PACKAGE/run_validation.sh"
```

它会真实使用两张 GPU，默认每卡分配器上限 40 GiB，并检查额外空闲空间。
16 张 H100 或多机环境仍须在目标硬件检查 Ray/NCCL 通信后再开始正式训练。

## 5. 启动正式训练和评测

具体命令见 README.md。继续保持上述 `PI05_ROOT`、`PACKAGE`、两个 Python 路径的
环境变量即可。不要把默认 global batch 100 原样用于 16 卡；例如 16 ranks × 每卡
microbatch 2 × 累积 4 = global batch 128：

```bash
export PI05_GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash "$PACKAGE/run_train.sh" --experiment a60 --output "$PI05_ROOT/training_runs/a60_epoch50" \
  --micro-batch-size 2 --gradient-accumulation-steps 4
```

多机时 `--gpus` 是 RLinf 的全局 GPU 索引，还需要已有 Ray 集群和相应
`--num-nodes`、`--ray-address` 参数。上面的命令不会替你创建多机集群。

## 验证记录

2026-10-29：两份清单都通过 Linux/Python 3.12 的 `uv pip compile` 完整依赖解析；
`verify_environment.py --resources` 分别在 A800 已跑通的两个环境通过，包括训练
模块与评测模块导入、关键版本、源码提交和资源检查。没有改动这两个现有环境，
没有为本次依赖文件重装完整环境或启动训练。此前同版本环境的短训练、断点恢复和
原始/A/B 共 12 条真实环境小评测已经完成。新机器的安装、驱动、渲染与多卡通信
需要在该机器执行检查和短验证后确认。

## 多卡与多机评测

训练的 `PI05_GPUS` 与评测的 `PI05_EVAL_GPUS` 独立配置。训练在同一模型的各 rank
之间同步梯度；评测每张卡各放一个模型副本，按 `(suite, task, trial)` 划分轨迹，
不做梯度同步。每个模型的完整评测总数仍为 2,000 条，不随卡数增加。

单机多卡，原始/A/B 三个模型依次占用这些卡：

```bash
export PI05_EVAL_GPUS=0,1,2,3,4,5,6,7
export PI05_EVAL_OUTPUT="$PI05_ROOT/evaluations/pi05_ab_full"
export PI05_A_CHECKPOINT=/absolute/path/to/a/checkpoints/global_step_N/actor
export PI05_B_CHECKPOINT=/absolute/path/to/b/checkpoints/global_step_M/actor
bash "$PACKAGE/evaluate_three.sh"
```

也可以只评测一个模型：

```bash
bash "$PACKAGE/run_eval_parallel.sh" --gpus 0,1,2,3 --label fewshot \
  --checkpoint "$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT" \
  --output "$PI05_ROOT/evaluations/fewshot_parallel"
```

续跑在同一条命令后加 `--resume`，保持任务、初始化状态、推理参数和全局分片数
不变。会重新校验权重哈希，已完成轨迹不重跑。输出下的 `workers/worker_NNN/`
保存各进程日志、视频和结果；全部完成后，根目录 `episodes/`、`summary.json`
和 `status.json` 与单卡输出格式一致，继续兼容 `compare_results.py`。

两台机器、每台 8 卡时，使用相同的共享输出目录、相同的源码/模型/资源路径，
每台机器各执行一条命令。例如评测原始模型：

```bash
# 节点 0
bash "$PACKAGE/run_eval_parallel.sh" --gpus 0,1,2,3,4,5,6,7 \
  --num-shards 16 --shard-offset 0 --label fewshot \
  --checkpoint "$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT" \
  --output /shared/evaluations/fewshot

# 节点 1（在另一台机器执行）
bash "$PACKAGE/run_eval_parallel.sh" --gpus 0,1,2,3,4,5,6,7 \
  --num-shards 16 --shard-offset 8 --label fewshot \
  --checkpoint "$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT" \
  --output /shared/evaluations/fewshot
```

评测无需 Ray/NCCL 集群；两个节点独立执行各自的分片。共享文件系统须支持文件锁，
防止两个启动器重复领取同一分片。先结束的节点会报告等待剩余分片，最后结束的
节点负责全局汇总。多机仍需在目标硬件验证路径共享、GPU/EGL 可见性与文件锁。

四条真实轨迹的多卡/续跑验证入口：

```bash
PI05_VALIDATION_GPUS=0,1 bash "$PACKAGE/run_parallel_validation.sh"
```

2026-10-29 已在 A800 GPU 1、3 完成此验证：两个独立进程各执行两条，共四条
真实轨迹和四段视频；汇总无重复，`--resume` 后所有已完成结果与视频的内容、
修改时间均保持不变。CPU 检查覆盖完整 2,000 条在 1/2/3/16/64/2048 个分片下的
分配，且会拒绝缺失轨迹、不同权重或重复领取分片的输出。实际 16-H100/多机硬件
执行尚未验证。

同一台 A800 上两个启动器并发领取不同全局分片、共同续跑和汇总也已通过；
合并输出与原有三模型比较脚本兼容。分卡后保持相同初始状态、seed 和任务步数
上限；跨 GPU 的动作数值可能存在差异，不以逐动作哈希完全一致作为通过条件。
