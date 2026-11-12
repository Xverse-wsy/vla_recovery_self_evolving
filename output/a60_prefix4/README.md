# Prefix4 的训练与评测输出

本目录提供 `a60_prefix4` 完整运行的数值记录和仿真视频。训练完成 50 轮、4,050 次更新；评测成功 1,495/2,000（74.75%）。配方、对照和比较限制见 [结果说明](../../results/a60_prefix4/README.md)。

| 文件 | 内容 |
| --- | --- |
| [training/metrics.jsonl](training/metrics.jsonl) | 4,050 次更新的 loss、梯度范数、学习率、样本累计数和完成轮次 |
| [training/status.json](training/status.json) | 完整训练状态和预算 |
| [evaluation/episodes.jsonl](evaluation/episodes.jsonl) | 2,000 条唯一任务/trial 的成功标记、步数、初始状态哈希和视频哈希 |
| [evaluation/videos.zip](evaluation/videos.zip) | 与逐条记录对应的 2,000 段仿真视频，使用 Git LFS 保存 |

训练更新编号从 1 开始，映射原指标事件的 0 起始编号。每个 suite 有 500 条评测；每条记录的 `video_file` 是 ZIP 内的相对路径，`video_sha256` 可用于校验视频。

```bash
git lfs install
git lfs pull --include="output/a60_prefix4/evaluation/videos.zip"
```
