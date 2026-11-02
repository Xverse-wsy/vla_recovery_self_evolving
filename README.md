# π0.5 LIBERO 训练与评测记录

本仓库提供 A60/B80 训练适配器，以及基于局部片段匹配和成功恢复后缀的 Stage1/Stage2 实验入口。本文汇总已完成实验的成绩、数据配方和操作入口；详细证据与待验证原因见 [RESULTS_AND_DIAGNOSIS.md](RESULTS_AND_DIAGNOSIS.md)。

## 已完成的评测结果

每个模型均完成四个标准 LIBERO suite、每个 suite 10 个任务、每任务 50 个预设初始状态，共 2,000 条评测。Baseline、A60、B80 的既有成绩保留为固定参考值，不安排复评；Stage1 使用本次完成的评测结果。

| 参考排名 | 模型 | Spatial | Object | Goal | LIBERO-10 | 总成功数 | 总成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A60 | 87.20% | 98.40% | 85.60% | 32.60% | 1,519 / 2,000 | **75.95%** |
| 2 | Baseline | 85.00% | 99.20% | 83.60% | 29.00% | 1,484 / 2,000 | **74.20%** |
| 3 | Stage1 | 75.20% | 86.60% | 76.00% | 15.80% | 1,268 / 2,000 | **63.40%** |
| 4 | B80 | 71.80% | 90.80% | 63.20% | 15.20% | 1,205 / 2,000 | **60.25%** |

Stage1 比固定 B80 参考值高 3.15 个百分点，比 Baseline 低 10.80 个百分点，比 A60 低 12.55 个百分点。本轮未达到超过 A60 的目标。

**比较口径：** A60/B80 的历史评测加载流程存在部分 FP32 参数经 BF16 中转产生舍入的问题；Stage1 已使用 dtype 配置后重新加载完整权重的流程。表格是固定历史参考值与新结果的对照，不是统一加载实现下的受控排名。舍入对历史成功率的影响未测量；Stage1 的训练初始状态与评测初始状态重合情况尚未核实。

## 数据配方与训练状态

| 实验 | 数据 | 训练样本或窗口 | 训练情况 |
| --- | --- | ---: | --- |
| Baseline | 原始 π0.5 LIBERO SFT 模型 | — | 本次不再训练 |
| A60 | 专家 40 条＋自主成功 20 条 | 9,864 帧 | 全参数训练 50 epoch，3,900 次更新 |
| B80 | A60 相同数据＋恢复成功 20 条 | 53,399 帧 | 全参数训练 50 epoch，20,900 次更新 |
| Stage1 | 专家、自主成功、中途 VLA 正向片段和准入 CAP 成功后缀 | 16,324 个窗口 | 冻结 VLM，纯正向 FM 训练 50 epoch，12,800 次更新 |

A60、B80 独立从原始模型初始化，实际实验使用 global batch 128，每轮完整遍历所有真实帧，沿用原模型归一化统计。仓库默认配置与已完成运行的参数不一定相同，复现实验前应核对最终配置。

Stage1 的窗口构成为专家 6,632、自主成功 3,232、中途 VLA 4,727、CAP 成功后缀 1,733；全部用于训练。匹配步长为 10，训练起点逐帧滑动。中途 VLA 片段只保留完整动作窗口，专家、自主成功和 CAP 成功尾部允许重复末动作监督。训练和完整评测均已完成。

Stage2 已有独立代码入口，本记录不提供其正式长训或完整评测成绩。短验证不能替代完整实验结果。

## 操作入口

接手项目先读 [AGENTS.md](AGENTS.md)。数据、模型、环境与输出目录由本地配置提供，运行产物和私有配置不随本记录提交。

| 工作 | 入口 |
| --- | --- |
| 安装训练与独立评测环境 | [ENVIRONMENT.md](ENVIRONMENT.md) |
| A60/B80 参数、执行顺序与续跑 | [RUNNING.md](RUNNING.md) |
| 局部片段匹配的数据要求 | [片段实验 README](pi05_recovery_pair/README.md) |
| Stage1 训练定义 | [README_STAGE1.md](pi05_recovery_pair/README_STAGE1.md) |
| Stage1 准备、训练与导出 | [RUN_POSITIVE_FM.md](pi05_recovery_pair/RUN_POSITIVE_FM.md) |
| 独立 Stage2 流程及 Stage1/Stage2 评测入口 | [README_STAGE2.md](pi05_recovery_pair/README_STAGE2.md) |
| 已有验证范围 | [VALIDATION.md](pi05_recovery_pair/VALIDATION.md) |

A60/B80 从仓库根目录执行。先复制并填写本地配置，再检查计划；正式入口会启动长训与完整评测：

```bash
cp experiment.env.example experiment.env
# 按 RUNNING.md 填写本地资源与输出配置。
bash run_experiment.sh plan
bash run_experiment.sh all --smoke
# 短验证通过并确认正式运行范围后执行。
bash run_experiment.sh all
```

已有运行可用 `bash run_experiment.sh all --resume` 续跑，保持数据、batch、并行配置和输出目录一致。单独训练或评测使用 `train` / `eval` 子命令；上面的 A60/B80 总入口不会自动运行 Stage1。

Stage1 使用 `pi05_recovery_pair/scripts/run_positive_fm_pipeline.sh prepare|train|export` 分阶段执行，完整评测使用独立评测入口。`stage2_evaluate_parallel.py --label stage1` 支持 Stage1 导出权重，具体环境和必需参数见 Stage2 操作文档；名称中的 Stage2 不表示会启动 Stage2 训练。

## 评测与后续排查

已完成评测使用 seed 0、3 个 flow steps、预测窗口长度 10、每次执行 5 个动作。Spatial、Object、Goal、LIBERO-10 的步数上限分别为 240、240、320、512，均不启用 WCM 或 CAP-X 在线恢复。完整汇总要求所有分片覆盖全部任务与初始状态，不能将部分完成的结果当作最终成绩。

后续关注 Stage1 的数据构成、训练预算和可训练参数范围，实验建议与事实依据见结果记录。当前没有证据证明单一因素导致全部性能差距，也没有启动新的训练或对照组复评。
