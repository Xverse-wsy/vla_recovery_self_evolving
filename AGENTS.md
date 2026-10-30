# Agent 接手入口

先读本文件，再按顺序读 `ENVIRONMENT.md`（部署）、`RUNNING.md`（配置和命令）。
这三份文件与仓库代码共同构成操作说明，不需要查阅此前对话。

## 独立的恢复片段匹配实验

新增 `pi05_recovery_pair/` 是独立的四类片段建库、状态匹配、成功片段后缀入池及
FM/pair训练实现。处理该目录时先读其 `AGENTS.md`、`README.md`、`DATA_CONTRACT.md`、
`VALIDATION.md`，按该目录的配置和命令运行；本文件以下A60/B80实验定义适用于根目录。
新增模块可复用本仓库训练环境，但pyproject仅列新增依赖，不能替代完整π0.5环境。
当前已在A800完成建库、视觉及成功后缀入池验证；全部真实负窗口配对、真实pair训练和
匹配准确率尚未完成。数据、模型、图像缓存、已安装环境和私有凭据不随代码提交。
最新负匹配仅使用VLA失败片段；CAP失败仍保留在来源库中，具体规则见子目录说明。

## 实验定义

- A / `a60`：原始专家40条＋VLA自主成功20条。
- B / `b80`：相同60条＋转换后的恢复成功20条。
- A、B必须各自从原始 `RLinf/RLinf-Pi05-LIBERO-SFT` 权重初始化，不能从A初始化B。
- 两组各完整遍历全部帧50遍。每帧是一个训练样本，不是每条轨迹算一个batch。
- 每个模型评测标准LIBERO的Spatial、Object、Goal、10：每suite10任务，
  每任务50个预设初始状态，共2,000条。评测原始模型、A、B，生成对照表。
- 评测不启用WCM或CAP-X。恢复数据的VLA前缀与CAP-X后缀已统一为7维OSC。
- 本仓库直接读取现有Parquet，不重新合并数据，不额外翻转图片，不重算归一化。
- 新增的失败恢复数据集是独立数据；当前A/B配方不会自动读取它。

## 部署与最短执行路径

1. 按 `ENVIRONMENT.md` 准备Linux x86_64、Python3.12、NVIDIA驱动和系统库。
2. 准备固定提交的RLinf/OpenPI源码、原始权重、三个完整数据目录、tokenizer、
   LIBERO的assets/bddl_files/init_files。它们不包含在本代码仓库中。
3. 创建两个独立环境，分别运行 `install_environment.sh train` / `eval`。
4. 复制 `experiment.env.example` 为 `experiment.env`，填写本机绝对路径和GPU。
5. 配置和检查资源，再预览计划：

```bash
set -a
source experiment.env
set +a
export PACKAGE="$PWD"
"$PI05_EVAL_PYTHON" configure_resources.py --resources "$PI05_ROOT/libero-resources"
"$PI05_TRAIN_PYTHON" verify_environment.py --mode train --resources
"$PI05_EVAL_PYTHON" verify_environment.py --mode eval --resources
bash run_experiment.sh plan
```

6. 用户允许使用选定GPU后，先执行短验证：

```bash
bash run_experiment.sh all --smoke
```

7. 仅在用户授权正式实验后，执行 `bash run_experiment.sh all`。
   也可分开 `train` / `eval`；中断续跑用 `all --resume`。

不要在仅要求部署或检查时启动正式50-epoch训练/6,000条评测。
不要停止别人的进程。实际启动前检查 `nvidia-smi`，显存检查不等于GPU资源预留。

## 参数与代码地图

| 修改内容 | 位置 |
| --- | --- |
| 数据/模型/环境路径、卡池、batch、输出目录 | `experiment.env` |
| 共享默认路径 | `config/paths.yaml` |
| epoch、学习率、保存间隔、保留数量、模型设置 | `config/base.yaml` |
| A/B来源及数量 | `config/a60.yaml`、`config/b80.yaml` |
| 总流程、最终模型交接、计划校验、阶段续跑 | `experiment.py`、`run_experiment.sh` |
| 训练入口、CLI覆盖、原始初始化/续跑审计 | `train.py`、`run_train.sh` |
| 精确epoch、采样、梯度/优化器/随机状态恢复 | `epoch_plan.py`、`training_runtime.py` |
| 原位读取、图像解码、动作窗口、OpenPI变换 | `direct_data.py`、`direct_loader.py` |
| 仿真/模型推理与评测参数 | `evaluate.py`、`run_eval.sh` |
| 多卡/多机分片、锁、完整性检查与汇总 | `evaluate_parallel.py`、`eval_jobs.py` |
| 三模型比较 | `compare_results.py` |

总batch＝全局训练rank数×每卡microbatch×梯度累积。16×2×4＝128。
若设置 `PI05_GLOBAL_BATCH`，清空 `PI05_ACCUMULATION`，或确保二者一致。
评测每卡一个模型副本和一个环境，GPU数控制并发，不使用训练batch。

当前总入口刻意固定A60/B80、50epoch及三模型完整评测。
如果增加数据源/改变实验规模，必须同时检查 `settings.py`、`train.py`、
`experiment.py` 的实验名/数量/epoch断言、短验证采样点及比较脚本；仅修改YAML
不能保证支持新实验。不要为通过断言而伪造数量或成功标签。

## 输出和续跑

`PI05_EXPERIMENT_OUTPUT` 下：

```text
experiment.json                 # 固定本次计划；改变参数请用新的输出目录
pipeline.log                    # 总日志
status.json                     # 总流程状态；train-only完成不等于评测也完成
training/{a60,b80}/
  status.json                   # 训练完成状态和最终checkpoint路径
  resolved_config.yaml
  dataset_manifest.json
  initial_weights.json          # 更新前权重审计；续跑不覆盖
  resumed_weights.json          # 若发生续跑
  direct_pi05/checkpoints/global_step_N/
    COMPLETE.json               # 完整保存标记
    actor/                      # 评测必须传actor目录
evaluation/{fewshot,a60,b80}/
  workers/worker_NNN/            # 分卡日志、视频、结果
  episodes/                     # 合并后的轨迹结果
  summary.json
comparison/comparison.md
```

短验证使用同一输出根下的 `smoke/`。若短验证训练中断，换新输出目录；正式训练
可续跑最近完整checkpoint。保持数据、batch、rank数、配置和输出目录不变。
评测续跑保持分片数不变，只跳过通过设置/权重校验的已完成轨迹。

## 多机边界和验证范围

- 单机多卡训练及评测已在A800双卡验证；A/B短训、checkpoint恢复和三模型仿真
  评测已执行。总入口控制流、共享路径、计划和中断恢复也有CPU验证。
- 依赖清单已解析、现有两个环境已核对；未在全新机器完整重装验证。
- 实际16张H100、多节点Ray/NCCL、共享文件锁仍须在目标机器验证。
- 多机训练要求已有Ray集群及各节点一致的资源路径；总脚本不会创建集群。
- 总入口自动评测使用主节点的本机卡池。跨节点评测须按 `ENVIRONMENT.md`
  分别启动每节点分片入口，共享输出；不能每节点各跑一遍完整评测。
- 本仓库不包含CAP-X转换工具链或其原始仿真日志。新恢复数据转换不能仅凭本仓库
  重建；训练使用已经转换并验证的数据。

## 常见失败的处理

- 路径/缺文件：先核对 `experiment.env`、原始40条的 `episodes.json`、完整`meta/`、
  tokenizer和LIBERO资源；不要删除稀疏episode编号或擅自重排动作。
- OpenPI/Transformers导入失败：使用训练安装脚本，确保最后重新安装fork版
  `rlinf-transformer-openpi`；不要将两个requirements混装。
- 显存不足：减少每卡microbatch、增加梯度累积；保留系统/CUDA余量，不能kill他人。
- EGL失败：检查驱动、EGL系统库、选定GPU及资源配置；`--cuda`通过不代表渲染通过。
- 旧输出拒绝新计划：用新输出目录；不要手改 `experiment.json` 绕过检查。
- checkpoint不完整：只使用含 `COMPLETE.json` 的checkpoint；不要手建标记文件。
- 性能指标：短验证只证明流程可运行，不证明完整准确率或跨GPU逐动作一致。

任何修改后至少做Python/Bash语法与 `git diff --check`，再做与修改相关的验证。
报告实际通过的验证，不把模拟后端验证写成真实GPU长训结果。
