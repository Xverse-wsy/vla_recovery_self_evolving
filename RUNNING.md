# 修改参数与启动实验

先按 ENVIRONMENT.md 安装两个 Python 3.12 环境，准备源码、权重、数据和 LIBERO
资源。下面的正式命令会启动长训/完整评测，请先执行 `plan` 检查配置。

## 统一配置

在代码仓库根目录：

```bash
cp experiment.env.example experiment.env
```

编辑 `experiment.env`（这个本机配置不会被 Git 提交）：

| 参数 | 用途 |
| --- | --- |
| `PI05_ROOT` | 工作目录，包含 RLinf/OpenPI/仿真资源等 |
| `PI05_TRAIN_PYTHON` / `PI05_EVAL_PYTHON` | 两个环境的 Python 绝对路径 |
| `PI05_BASE_CHECKPOINT` | 原始 RLinf few-shot π0.5 权重目录；A、B 都从它初始化 |
| `PI05_NORM_STATS` | 原模型的归一化统计文件 |
| `PI05_AUTHOR40_DIR` | 原始 40 条目录，保留 episodes.json |
| `PI05_AUTONOMOUS20_DIR` | 自主成功 20 条目录 |
| `PI05_RECOVERY20_DIR` | 转换后的恢复成功 20 条目录 |
| `PI05_GPUS` | 训练使用的物理/全局 GPU 索引 |
| `PI05_MICRO_BATCH` | 每张 GPU 每个前向/反向步骤的样本数 |
| `PI05_ACCUMULATION` | 每次参数更新前累积多少个步骤 |
| `PI05_GLOBAL_BATCH` | 可选；直接指定总 batch 时清空 ACCUMULATION |
| `PI05_GPU_MEMORY_LIMIT_GIB` / `PI05_GPU_HEADROOM_GIB` | 分配器上限及额外空闲空间要求 |
| `PI05_EVAL_GPUS` | 评测使用的本机 GPU 索引 |
| `PI05_EXPERIMENT_OUTPUT` | 本次实验的独立输出根目录 |

总 batch = 训练 GPU 数 × 每卡 microbatch × 梯度累积。
例如 16 × 2 × 4 = 128。提高 microbatch 会增加单卡显存需求；提高累积次数通常
主要增加每次更新的计算时间。更新步数会根据 batch 重新计算，仍完整遍历数据
50 次；轨迹尾部与 epoch 尾部会正确处理。

也可以直接编辑 `config/paths.yaml` 设置路径、`config/base.yaml` 设置默认 batch、
学习率和 checkpoint 策略。环境变量的覆盖优先级更高。训练、评测的来源统计与
环境检查共享这份路径配置，不用再修改 evaluate.py。

## 总入口

```bash
# 仅打印路径、数据量、batch、更新次数和训练命令；不加载模型、不使用 GPU。
bash run_experiment.sh plan

# A：40+20；B：相同的40+20再加恢复20。两组各训练50 epoch。
bash run_experiment.sh train

# 读取上面两个训练的最终 checkpoint，评测原始/A/B，各2000条，并生成表格。
bash run_experiment.sh eval

# 连续完成上述训练和评测。
bash run_experiment.sh all

# 正式训练中断后，自动找最近完整 checkpoint；已完成的训练和评测会跳过。
bash run_experiment.sh all --resume
```

可先做短验证（两组各 15 帧、2 epoch，三个模型各 4 条真实评测），输出在配置
目录下的 `smoke/` 子目录：

```bash
bash run_experiment.sh all --smoke
```

短验证的训练阶段若中断，使用新的输出目录重新验证；正式训练支持 checkpoint
续跑。运行中的实验不要更改数据、batch、GPU 数或其他计划参数；要更改时使用
新的输出目录，总入口会拒绝混用不同计划。

日志在 `$PI05_EXPERIMENT_OUTPUT/pipeline.log`，最终对照表在
`$PI05_EXPERIMENT_OUTPUT/comparison/comparison.md`，完整指标在同目录的 JSON。
该总入口的 A/B 训练依次执行，使用相同训练卡池；三个模型的评测也依次使用配置
的本机评测卡池。每个模型的轨迹在这些卡之间划分，合计仍为 2,000 条。

后台运行示例（先完成 plan 和短验证）：

```bash
screen -dmS pi05_ablation bash -lc 'cd /absolute/path/to/pi05_ablation && bash run_experiment.sh all'
screen -r pi05_ablation
```

## 单独启动底层脚本

如果希望自己安排阶段，在当前 shell 加载配置后运行：

```bash
source experiment.env
export PACKAGE="$PWD"

bash run_train.sh --experiment a60 --output "$PI05_ROOT/training_runs/a60_custom" \
  --micro-batch-size 2 --gradient-accumulation-steps 4

bash run_train.sh --experiment b80 --output "$PI05_ROOT/training_runs/b80_custom" \
  --micro-batch-size 2 --gradient-accumulation-steps 4
```

训练 CLI 参数覆盖 YAML 默认值。直接调用 `run_train.sh` 时，`PI05_MICRO_BATCH`
和 `PI05_ACCUMULATION` 是总入口配置，不会自动转成 CLI；请显式传上面的参数。

单独评测三个模型，两个 actor 路径从各自训练 `status.json` 的 `checkpoint` 后加
`/actor` 得到：

```bash
export PI05_A_CHECKPOINT=/absolute/path/to/a/checkpoints/global_step_N/actor
export PI05_B_CHECKPOINT=/absolute/path/to/b/checkpoints/global_step_M/actor
export PI05_EVAL_OUTPUT="$PI05_ROOT/evaluations/three_models"
bash evaluate_three.sh
```

评测是每卡独立执行轨迹，不需要训练式的全局 batch/梯度累积。当前每卡一个环境，
用卡数控制并发。跨节点评测的分片命令见 ENVIRONMENT.md。

## 两台8卡机器的训练

总入口只在一个主节点启动。先准备好现有 Ray 集群、两个节点一致的源码/权重/
数据/输出路径和共享文件系统，然后在配置中设置：

```bash
export PI05_GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PI05_TRAIN_NUM_NODES=2
export PI05_RAY_ADDRESS=auto  # 或实际的 Ray 地址
export PI05_EVAL_GPUS=0,1,2,3,4,5,6,7
```

训练会交给已有 Ray 集群调度；总入口的评测阶段使用主节点这8张卡完成全部评测。
需要评测也跨节点时，使用 ENVIRONMENT.md 中的分片入口。实际16-H100/多机硬件
仍须先做短验证。
