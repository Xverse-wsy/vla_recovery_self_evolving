# 按原始执行结果均衡选择失败轨迹

**2026-11-02 用户已确认：本次采用v2的20条（LIBERO-10 10＋Goal 8＋Spatial 2），
只来自指定400条评测。不是18条方案，不使用混入其他批次的v3。**
A800清单目录：
`/home/wenchaoxu/wsy/vla-recovery-posttrain-loop/dataset_subsets/pi05_original_failure_balanced20_20261102_v2/`。
`CURRENT_SELECTION.json` 保存本次确认和选择清单哈希。

用户随后确认将成功20按3/5/10/2整理，并与本失败20分别保存成完整原始轨迹包。
见[方案1原始数据整理说明](README_RECOVERY_RAW_SCHEME1.md)；本失败清单原样保留，未重新抽样。

只读取原始评测的 `report.json`：`success=false` 且 `used_capx=true`。
不检查转换后的成功率，不要求重放仍失败，不启动仿真或恢复验证。
缺失/未知结果不能标成失败。

任务按 suite 和公开任务指令分组，先每任务选一条。覆盖全部可用任务后，
优先补当前入选数量最少且还有候选的suite；同suite内优先选已入选次数最少的任务。
因此少数suite可能有一个任务选到3条，而多数suite的部分任务仍只有1条。
同任务优先选择原始记录完整的轨迹；这只检查文件，不重新执行。固定 seed 可复现。
CAP 启动失败且没有实际 CAP 动作的条目单列为 `cap_launch_failed_vla_only`，
不能作为已执行 CAP 的失败动作片段。原始保存不完整的条目也单列字段。

在仓库根目录运行（Python 标准库即可）：

```bash
python select_failure_episodes.py \
  --results /path/to/original/eval/results \
  --output /path/to/new_failure_selection \
  --count 20 --seed 42
```

输出 `selection.json`、`selected.csv`、`task_distribution.csv` 和完整失败候选清单。
代码允许重复 `--results`，但**本次只允许那400条评测对应的单个results目录**，
不得混入其他批次。扫描其中全部40任务，不限定到先前选中任务；相同原始路径只计一次。
这是带来源路径与原报告哈希的**选择清单**，不是转换后的可训练数据集。
如只需要每任务一条，本次候选有13个任务，可用新输出目录指定 `--count 13`。

2026-11-02 A800 原始失败池：54条、13个任务（LIBERO-10 44、Goal 8、Spatial 2）。
当前v2均衡20条：13任务、每任务1～3条（LIBERO-10 10、Goal 8、Spatial 2）；
Goal的“open the top drawer and put the bowl inside”有3条，其余任务最多2条。
v1为11/7/2，单独保留；v2在完整任务覆盖后优先补少数suite。
Object没有原始失败。选中两条Goal中抽屉任务仅有VLA前缀，CAP启动失败，记录不完整。
曾另外生成探索性v3清单，混入早期first10批次的1条Spatial失败，9/8/3；
**该v3不符合用户随后明确的“仅那400条”范围，不用于本次抽样或训练。**
当前有效的20条清单为v2：10/8/2；此前提出的18条方案未采用。
原有 failure20、切分库以及 Stage1/Stage2 输入不会被本脚本自动替换。
