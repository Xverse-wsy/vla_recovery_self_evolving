# 方案1：重新整理原始恢复成功20条和失败20条

本次只整理指定400条评测中的原始轨迹及来源清单，不转换OSC、不回放、不切分、不启动训练。
原始报告的 `used_capx=true` 且 `success=true/false` 区分两组。
成功只表示episode最终完成，不表示全部恢复尝试或动作均成功。

| Suite | 成功20 | 已确认失败20 | 合计40 |
|---|---:|---:|---:|
| LIBERO-10 | 3 | 10 | 13 |
| Goal | 5 | 8 | 13 |
| Spatial | 10 | 2 | 12 |
| Object | 2 | 0 | 2 |

成功组优先覆盖不同任务（本批候选可覆盖16个任务），同任务重复时优先平衡成功＋失败的任务数量。
固定seed42，不额外按最长/最短轨迹筛选，不用历史1200步限制。
失败20必须使用用户已确认的v2清单及 `CURRENT_SELECTION.json`，原清单哈希保持不变。
其中两条Goal是CAP启动失败，原轨迹记录不完整；它们仍按用户确认保留为原始失败案例，不能伪称存在CAP动作。

脚本只使用Python标准库。A800复现命令：

```bash
python3 prepare_balanced_recovery_raw.py \
  --results /home/wenchaoxu/wsy/vla-recovery-posttrain-loop/wcm_pi05_monitor/outputs/hybrid_independent_gpu1_20261019_172047/eval/results \
  --failure-selection /home/wenchaoxu/wsy/vla-recovery-posttrain-loop/dataset_subsets/pi05_original_failure_balanced20_20261102_v2/selection.json \
  --output /home/wenchaoxu/wsy/vla-recovery-posttrain-loop/dataset_subsets/pi05_recovery_raw_scheme1_20261102_v1 \
  --seed 42 --storage hardlink
```

输出目录必须全新；`.building`用于未完成的临时输出，全部通过审计后原子改名为最终目录。
同一文件系统默认硬链接大于等于1MiB的原始文件，小文件复制并核对SHA256；也可指定`--storage copy`独立复制。
逐一核对原始轨迹manifest中所有NPZ分片的SHA256、分片连续性及记录长度。
这是文件完整性核对，不是轨迹回放或恢复成功确认，不调用模型/API或GPU。

输出结构：

```text
pi05_recovery_raw_scheme1_20261102_v1/
  recovery_success20_raw/
    episodes/episode_000_.../      # 原report、trajectory、WCM、CAP日志和原视频
    selection.json
    episodes.jsonl
    episodes.csv
  recovery_failure20_raw/
    episodes/episode_000_.../
    selection.json                # 已确认失败清单，原样复制
    CURRENT_SELECTION.json
    episodes.jsonl
    episodes.csv
  summary.json                    # 实际suite/任务/步数统计；train_ready=false
  trajectory_integrity.json       # 每条轨迹实际保存步数及分片核对结果
  files.jsonl                     # 逐文件来源及复制/硬链接方式
  status.json
  README.md
```

`snapshot_relative_path`相对于本包根目录，迁移应带完整目录内容。原始日志中的A800绝对路径原样保留作溯源，
后续读取整理后的文件需从本包相对路径定位。不要编辑硬链接文件；删除原目录不会使快照文件消失。
此输出尚无LeRobot训练数据或新的成功代码段标签，OSC转换和按程序切分留到下一步。
旧成功20、失败20、切分库及现有训练配置不被本脚本替换。

## 2026-11-02 实际整理结果

上述A800输出目录已完成。成功20覆盖16任务，失败20覆盖13任务，合计覆盖24个不同任务。

| 原始执行步数 | 成功20 | 失败20 | 合计40 |
|---|---:|---:|---:|
| VLA | 2,265 | 2,337 | 4,602 |
| CAP | 23,582 | 71,585 | 95,167 |
| 全部 | 25,847 | 73,922 | 99,769 |

这些是原始执行步数，包含失败尝试/停滞等，不能等同于正向监督帧数或最后成功程序的长度。
共10,095个原始文件，逻辑大小37,110,448,209字节；大文件硬链接6,673个。
全部6,255个动作分片通过原始SHA256检查。10项CPU测试通过，包含完整打包和保留不完整失败记录的回归。
没有执行OSC转换、轨迹回放、按程序切分、模型训练或LIBERO评测。
