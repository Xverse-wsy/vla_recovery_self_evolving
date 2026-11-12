# π0.5 LIBERO 训练与评测记录

本仓库提供训练与评测适配代码，并保存已完成实验的结果。本次 A60＋4 条前缀已完成 50 轮训练和 2,000 条 LIBERO 评测，成功率为 **74.75%**；结果与比较限制见 [本次结果](results/a60_prefix4/README.md) 和 [汇总记录](RESULTS_AND_DIAGNOSIS.md)。

## 完整结果对照

每个模型均完成 2,000 条评测，Spatial、Object、Goal、LIBERO-10 各 500 条。下表按记录的总成功率排序，最后一列为相对 Baseline 的百分点差值。K 表示评测的 flow steps。

| 模型 | K | Spatial | Object | Goal | LIBERO-10 | 总成功率 | 相比 Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A60 | 3 | 87.20% | 98.40% | 85.60% | 32.60% | 75.95% | +1.75 |
| Prefix4 | 3 | 86.20% | 98.80% | 85.80% | 28.20% | 74.75% | +0.55 |
| Baseline | 3 | 85.00% | 99.20% | 83.60% | 29.00% | 74.20% | — |
| Stage1 方案 1（全参数） | 5 | 86.40% | 98.60% | 83.80% | 23.40% | 73.05% | -1.15 |
| A60＋22 个子片段 | 3 | 81.00% | 98.40% | 82.20% | 22.60% | 71.05% | -3.15 |
| 旧 Stage1（冻结 VLM） | 3 | 75.20% | 86.60% | 76.00% | 15.80% | 63.40% | -10.80 |
| B80 | 3 | 71.80% | 90.80% | 63.20% | 15.20% | 60.25% | -13.95 |

Prefix4 是目前新增片段实验中总成功率最高的版本：比 A60＋22 个子片段高 3.70 个百分点，比 Stage1 方案 1 高 1.70 个百分点，比 Baseline 高 0.55 个百分点，仍比 A60 低 1.20 个百分点。相对 A60，Object、Goal 略有提升，主要差距在 LIBERO-10；目前尚未隔离造成差距的因素。

Stage1 方案 1 使用 K=5，其余使用 K=3；历史 A60/B80 与新实验的权重加载方式不同。因此，这是已有成绩的描述性对照，不是完全同条件的受控排名。训练与评测初始状态是否重合仍未知。Prefix10 未纳入本表。

## 实验版本说明

| 模型 | 数据与训练方式 |
| --- | --- |
| Baseline | 原始模型，没有追加训练 |
| A60 | 40 条专家轨迹＋20 条自主成功轨迹，全参数训练 |
| B80 | A60 数据＋20 条完整恢复轨迹，全参数训练 |
| 旧 Stage1 | 专家、自主成功、筛选的推进片段和恢复成功后缀，冻结 VLM |
| Stage1 方案 1 | 专家、自主成功和筛选的 VLA 推进片段，全参数训练 |
| A60＋22 个子片段 | A60 数据＋22 个复核子片段，全参数训练 |
| Prefix4 | A60 数据＋4 条裁剪后的 VLA 前缀，全参数训练 |

## 本轮发布文件

| 目录 | 内容 |
| --- | --- |
| [results/a60_prefix4](results/a60_prefix4/README.md) | 本次成绩、配方、评测协议、逐任务排查与文件哈希 |
| [output/a60_prefix4](output/a60_prefix4/README.md) | 本次 4,050 条训练指标、2,000 条评测及对应视频 |
| [results/a60_subtasks22](results/a60_subtasks22/README.md) | 22 段实验的成绩、配方、协议和对照 |
| [output/a60_subtasks22](output/a60_subtasks22/README.md) | 22 段实验的 4,350 条训练指标、2,000 条评测及 2,000 段视频 |
| [results/stage1_scheme1](results/stage1_scheme1/README.md) | 新版汇总、训练配方、评测协议、历史对照及文件哈希 |
| [output/stage1_scheme1](output/stage1_scheme1/README.md) | 5,250 条训练指标、2,000 条逐条评测及 2,000 段视频 |
| [results/libero_comparison](results/libero_comparison/README.md) | 四个历史模型的 8,000 条记录及对应视频 |

视频使用 Git LFS 保存。取得本轮视频后，按 ZIP 内的相对路径查看；逐条结果中包含视频 SHA-256。

```bash
git lfs install
git lfs pull --include="output/a60_prefix4/evaluation/videos.zip"
```

## 新版 Stage1 配方

从原始 few-shot 模型初始化，全参数训练，使用原生 H10×32 FM 均值与 A60 配方的优化器设置。13,407 个训练窗口包括专家 6,632、自主成功 3,232、VLA 推进片段 3,543；严格 CAP 准入为 0。全部窗口用于训练，global batch 为 128，完整 50 轮共 5,250 次更新、670,350 次窗口呈现。旧 Stage1 权重未用作初始化。

中途 VLA 推进片段已删除末 10 步，训练仅纳入完整真实窗口；专家和完整自主成功轨迹尾部允许重复末动作。配方与实际完成状态分别见 [training.json](results/stage1_scheme1/training.json) 和 [status.json](output/stage1_scheme1/training/status.json)。

Stage1 方案 1 的训练使用提交 [`96c4fe8`](https://github.com/Xverse-wsy/vla_recovery_self_evolving/commit/96c4fe8815ac52a72fcf467dd018606f3eff82d9) 的代码，操作入口为该版本的 `pi05_recovery_pair/README_STAGE1.md`。本结果分支保留既有代码历史，复现新版实验需使用记录的代码版本。Stage2 旧入口与新版全参数 Stage1 不兼容，本轮未运行 Stage2。

## 比较限制

Stage1 方案 1 评测使用 5 个 flow steps，其他表列模型使用 3 个；其余任务范围、seed 0、长度为 10 的预测窗口、每次执行 5 个动作和归一化统计保持一致。新版与四个历史模型的任务、trial 和初始状态哈希逐条一致，但训练与评测初始状态重合情况未核实，不能声称独立留出集评测。

历史 A60/B80 完整权重加载涉及中间 dtype 舍入，新旧 Stage1、A60＋22 个子片段和 Prefix4 使用 dtype 配置后重载完整权重的流程。发布资料保留这项差异，不把历史成绩换算成尚未测量的修正结果。
