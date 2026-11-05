# 新版 Stage1 的训练与评测输出

本目录提供已完成方案 1 实验的脱敏输出，可复核训练预算、逐条评测结果和对应视频。成功率与比较限制见 [结果说明](../../results/stage1_scheme1/README.md)。

| 文件 | 内容 |
| --- | --- |
| `training/metrics.jsonl` | 5,250 条更新记录，保留 loss、梯度范数、学习率、epoch 与样本计数 |
| `training/status.json` | 完成 50 轮、5,250 次更新和 670,350 次窗口呈现的状态 |
| `evaluation/episodes.jsonl` | 2,000 条完整且无重复的评测记录及视频位置与哈希 |
| `evaluation/videos.zip` | 对应的 2,000 段 MP4 仿真视频，使用 Git LFS |

视频归档内按 `suite/task_XX/trial_YY.mp4` 排列；逐条结果的 `video_file` 相对于 ZIP 根目录。获取视频：

```bash
git lfs pull --include="output/stage1_scheme1/evaluation/videos.zip"
```

输出按字段白名单从原始文件提取。训练指标只保留数值和布尔值；评测只保留任务标识、结果、步数、必要哈希及归档内相对路径。运行日志、原始配置、checkpoint、绝对路径、账户、凭据、机器标识、部署信息、设备容量和耗时字段均不包含在本包中。

视频写入器直接记录仿真相机画面，不叠加终端内容。发布时检查每段视频的容器与流 metadata、单视频流、帧数、SHA-256 以及 ZIP CRC；另抽查代表帧。没有逐段人工观看或完整解码全部帧，不将抽查表述为全量画面审阅。
