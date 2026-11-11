# A60＋原始 22 段的训练与评测输出

本目录提供 `a60_subtasks22` 已完成实验的脱敏训练指标、逐条评测和仿真视频。成功率及历史比较限制见 [结果说明](../../results/a60_subtasks22/README.md)。

| 文件 | 内容 |
| --- | --- |
| `training/metrics.jsonl` | 4,350 条更新记录，仅包含更新编号、loss、梯度范数、学习率、样本数和已完成轮数 |
| `training/status.json` | 50 轮、4,350 次更新、552,900 次窗口呈现的完成状态 |
| `evaluation/episodes.jsonl` | 2,000 条逐条评测，以及视频相对位置、文件大小和 SHA-256 |
| `evaluation/videos.zip` | 对应的 2,000 段 MP4 仿真视频，使用 Git LFS |

训练指标从事件记录的指定数值字段重建，`update` 为从 1 开始的更新编号；原事件记录的 step 从 0 开始。原始事件文件及其文件名、运行日志、配置、权重和优化器状态均不包含在输出包中。

视频按 `suite/task_XX/trial_YY.mp4` 排列；逐条结果的 `video_file` 相对于 ZIP 根目录。获取视频：

```bash
git lfs pull --include="output/a60_subtasks22/evaluation/videos.zip"
```

仅保留复核实验所需的数值、任务标识与内容哈希。个人目录、账号、凭据、机器标识、内部地址、部署信息、设备型号与数量、容量、运行时间及归档原始时间戳均已排除。每段视频的容器和流 metadata、单视频流、帧数及内容哈希均经过检查，ZIP 使用统一固定时间戳。

视频写入代码只记录仿真相机画面，不叠加终端输出。另检查每个 suite 的成功和失败示例帧；这不等于人工查看全部视频帧。具体检查范围见 [脱敏记录](../../results/a60_subtasks22/sanitization.json)。
