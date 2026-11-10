# π0.5训练与评测适配

**新增实验：A60＋原始22个可靠子片段**。入口：[README_SUBTASKS22.md](README_SUBTASKS22.md)。
专家40＋自主成功20＋22段，共11058个密集窗口，完整50epoch，直接继承A60全参数SFT配方。
中间片段不补齐，最终LIBERO成功片段沿用成功尾部规则；prompt使用父轨迹instruction。
本次CPU验证，GPU测试等A800恢复。原方案1和A60/B80入口保留如下。

Agent先读 [AGENTS.md](AGENTS.md)。**当前片段Stage1的唯一操作入口：[README_STAGE1.md](pi05_recovery_pair/README_STAGE1.md)**。它写明阶段目标、依赖、数据目录、严格建库、FM训练、GPU限制、短测试、恢复和导出；不需要先前对话。

当前方案1使用原始专家40、自主成功20，以及新恢复成功/失败父轨迹中的VLA推进片段；严格CAP匹配当前准入0。共13,407个逐帧窗口、13,767去重动作。VLM不冻结，使用原生H10×32 FM及A60优化器/调度；global128时完整50轮共5,250更新。VLA推进片段先去掉末10步，再仅保留完整真实窗口，不补齐。

新机按 `recovery_conversion/build_scheme1.py --stage banks|match|prepare|align-training` 分阶段从已有HF数据重建，再用 `pi05_recovery_pair/scripts/run_positive_fm_pipeline.sh train|export`。来源工具边界见 [README_SCHEME1_STAGE1.md](recovery_conversion/README_SCHEME1_STAGE1.md)，实测见 [VALIDATION.md](pi05_recovery_pair/VALIDATION.md)。旧16,324窗口、14段CAP和冻结配方均为历史版本。

本轮仅做有界测试和代码推送，不启动正式50轮。下文是独立A60/B80对照实验，不能用其总入口启动当前Stage1。Stage2也有独立文档，旧入口尚未适配此版全参数Stage1 checkpoint。

This directory contains the external training/evaluation adapter. It reads the
existing three selected datasets directly; it does not create merged datasets
or modify the original datasets, RLinf source, or base model.

For a new Linux GPU machine, follow [ENVIRONMENT.md](ENVIRONMENT.md). Install
`requirements.txt` for training and `requirements-eval.txt` in a separate
evaluation environment using `install_environment.sh`.

For editable data paths, batch/GPU settings and a complete
train-A → train-B → evaluate-original/A/B pipeline, see [RUNNING.md](RUNNING.md).
Copy `experiment.env.example` to `experiment.env`, edit it, then run
`bash run_experiment.sh plan` before `bash run_experiment.sh all`.

## Experiments

| Name | Trajectories | Training frames | Initialization |
| --- | --- | --- | --- |
| `a60` | author40 + autonomous20 | 9,864 | original RLinf few-shot π0.5 |
| `b80` | same 60 + recovery20 | 53,399 | original RLinf few-shot π0.5 |

Both configurations run 50 complete epochs. Every real frame appears once per
epoch; action windows stop at episode boundaries. With the default global batch
100, A has 4,950 updates and B has 26,700 updates. The final update of an epoch
can be smaller. Both use the original checkpoint's normalization statistics.

## Environment on A800

```bash
export PI05_ROOT=/home/wenchaoxu/wsy/vla-recovery-posttrain-loop
export PACKAGE="$PI05_ROOT/training_configs/pi05_ablation_direct_20261029"
export PI05_TRAIN_PYTHON="$PI05_ROOT/environment-setup/.venv-pi05-sft/bin/python"
export PI05_EVAL_PYTHON=/home/wenchaoxu/miniconda3/envs/corrector-eval/bin/python
```

## Short validation

```bash
PI05_VALIDATION_GPUS=1,3 bash "$PACKAGE/run_validation.sh"
```

This checks the source reader against LeRobot, exact epoch sampling (including
16-rank CPU gradient references), two isolated 15-sample/two-epoch training runs,
checkpoint recovery, and original/A/B evaluation on one initial state in one
task per suite. These small evaluations verify execution, not model quality.
The final report is `validation_report.json` under the run directory referenced
by `latest_validation.txt`. Videos and episode records are saved there too.

## Formal training commands (not launched by validation)

```bash
export PI05_GPUS=1,3
bash "$PACKAGE/run_train.sh" --experiment a60 --output "$PI05_ROOT/training_runs/a60_epoch50"
bash "$PACKAGE/run_train.sh" --experiment b80 --output "$PI05_ROOT/training_runs/b80_epoch50"
```

Run A and B sequentially if they share GPUs. Specify unused output directories.
Microbatch and accumulation are configurable, for example:

```bash
bash "$PACKAGE/run_train.sh" --experiment a60 --output "$PI05_ROOT/training_runs/a60_custom" \
  --micro-batch-size 2 --gradient-accumulation-steps 25 \
  --gpu-memory-limit-gib 40 --gpu-headroom-gib 12
```

Global batch = number of ranks × microbatch × accumulation. Alternatively pass
`--global-batch-size` (must divide by ranks × microbatch). Epoch/update counts
are recalculated. Allocator limits do not include every CUDA/context allocation;
headroom is checked before loading. No launcher kills existing processes.

`--gpus` uses RLinf physical/global GPU indices. Multi-node runs additionally
require `--num-nodes` and an existing `--ray-address`. Only two-GPU execution has
been tested on A800; 16-H100 hardware placement and communication still need a
short validation on that hardware before formal training.

Save interval is 500 optimizer updates, with the latest three checkpoints kept
and a final checkpoint always saved. Resume with the same experiment, batch,
GPU count, data, and output directory plus
`--resume /absolute/path/to/checkpoints/global_step_N`.

## Full evaluation commands (not launched by validation)

Set `PI05_A_CHECKPOINT` and `PI05_B_CHECKPOINT` to the final checkpoint **actor**
directories, then run:

```bash
export PI05_EVAL_GPU=1
export PI05_EVAL_OUTPUT="$PI05_ROOT/evaluations/pi05_ab_full"
bash "$PACKAGE/evaluate_three.sh"
```

For automatic episode partitioning across GPUs, set `PI05_EVAL_GPUS` instead:

```bash
export PI05_EVAL_GPUS=0,1,2,3,4,5,6,7
bash "$PACKAGE/evaluate_three.sh"
```

Each GPU runs a separate model copy and a disjoint set of episodes. Each model
still evaluates 2,000 total episodes across all GPUs. The launcher pins both
inference and EGL rendering to the selected GPU, writes separate worker logs,
and merges only after checking complete coverage and matching checkpoint hashes.
Use `--resume` with the same shard count to continue. See ENVIRONMENT.md for
multi-node evaluation with a shared output directory.

Defaults evaluate all four standard LIBERO suites, all ten tasks, and all fifty
initial states: 2,000 episodes per model. WCM and CAP-X are disabled. Identical
seeds, initial states, and suite budgets are used across models. Reports mark
initial states shared with rollout training data. `--resume` skips completed
episodes after checking run settings. Videos are saved unless `--no-video` is
passed. `compare_results.py` checks matching episode identities and writes the
comparison reports.
