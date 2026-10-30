# 数据与证据合约

## Manifest

每行一个已有控制段，整个 parent episode 的段共享 episode_id。
路径应为绝对路径。所有帧／动作范围零起点、右端不包含。

已有切分集的 `selection.json` 是片段分类来源：vla_progress_candidate→D_vla_positive、
vla_wcm_warning→D_vla_negative、capx_completed→D_code_positive、
capx_not_completed→D_code_negative。自主成功集加入 D_vla_positive。
manifest 的 `bank_class` 保存该归属，`bank_label` 保存分类依据、原始类别、selection／
provenance 路径及 SHA256。类别互斥，并核对 owner、程序完成标志与原 provenance 一致。

这些片段直接建立四类索引，不重新判断原分类。按用户当前实验定义，`training_quality`
直接使用 `bank_class` 的正负；`local_quality=unknown` 不再阻止这些片段匹配或训练。
local_quality 与旧可信 mask 保留为可选细粒度证据，不伪造逐动作已复核声明。
负类片段没有额外细粒度标签时，以窗口有效动作作为 pair 错误范围；如有显式负动作证据，
使用其可信 mask。最终 pair mask 仍与正负有效 mask 相交并移除填充和共享前缀。
每个 pair 记录 negative_label_basis、negative_bank_label、error_mask_basis。
当前negative_query_policy=vla_failure_only_v1：负匹配查询仅取D_vla_negative。
D_code_negative的原分类与轨迹均保留，但不进入负查询、可用负训练池或pair采样。
正参考仍为VLA正窗口＋已纳入的CAP成功后缀。已有WCM错误片段直接作为本轮VLA负类，
不另按整条parent最终成败重新筛选，也不自动将一个匹配窗口的后续负片段整段纳入。
CAP 正窗口先对原 VLA 正库做起点匹配。当前preset的cap_success_admission=segment_suffix：
同一成功代码片段有任意窗口匹配后，从最早匹配窗口起点到该原片段结束的连续动作全部纳入。
前缀不纳入，不跨越原程序／episode／split／标签边界，也不扩展到失败程序。
后续窗口保留原动作与有效mask，窗口仍不跨片段边界。负窗口随后寻找严格正参考。
coverage记录directly_matched、suffix_anchor_id、suffix_anchor_reference和原始行范围；
后续窗口的reference_applies_to=suffix_anchor，不能声称该后续起点自身已经匹配VLA。
cap_success_suffixes.jsonl保存原Parquet行范围和父轨迹步数范围，右端不包含。
这是采样库索引的纳入规则，不重写原轨迹；动作payload仍直接读取已有归一化窗口。

```json
{
  "segment_id": "unique-segment-id",
  "parquet": "/absolute/path/episode_000001.parquet",
  "length": 12,
  "task_id": "libero_object/task03",
  "episode_id": "unique-original-parent-rollout",
  "initial_group_id": "verified-task-initial-state-hash",
  "initialization_verified": true,
  "event_id": "local-failure-or-program-event-id",
  "controller": "code",
  "instruction": "original task instruction",
  "parent_start_step": 133,
  "state_layout": "eef_xyz_axisangle_gripper2",
  "handoff_start": true,
  "event_starts": [0],
  "annotations": [
    {
      "start": 0,
      "end": 6,
      "local_quality": "positive",
      "quality_evidence": {
        "verified_local_actions": true,
        "reviewer": "reviewer-or-verified-detector-version",
        "evidence": "path/to/local-action-evidence"
      }
    },
    {
      "start": 6,
      "end": 12,
      "event_id": "failure-event-1",
      "local_quality": "negative",
      "quality_evidence": {
        "verified_local_actions": true,
        "negative_behavior_confirmed": true,
        "reviewer": "reviewer-or-verified-detector-version",
        "evidence": "path/to/local-action-evidence"
      }
    }
  ],
  "start_context": {
    "0": {
      "phase": "approach",
      "target_object": "bowl",
      "gripper_state": "open",
      "held_object": "empty",
      "contact_state": "none",
      "progress": 0.35,
      "progress_scorer_id": "frozen-scorer-and-calibration-sha256",
      "critical_state_verified": true,
      "evidence": {"source": "verified start-state log or image review"},
      "object_poses": {}
    }
  },
  "source": {
    "program": "/path/to/turn_03.py",
    "action_mapping": "/path/to/segment_mapping.jsonl"
  }
}
```

示例中的 start_context 是可选历史记录，不是当前匹配输入；为空也可匹配。
当前匹配只读取执行前的原任务指令、RGB和模型本体观测。若另行提供 annotations，
其 positive/negative 声明仍需行为证据；局部负区间中的所有动作
都被声明可信，若只有其中部分可信，应进一步拆分区间，其余 unknown。
标签范围不能重叠。程序边界、控制权边界由 manifest 段表示，不允许窗口跨段。

Parquet 必须包含 `state` float[8]、`actions` float[7]、`image`、`wrist_image`。
图像支持已有 `{bytes: PNG, path: null}` 字段，图像读出不再旋转。
state8 的顺序是 EEF xyz米、EEF axis-angle弧度、两维夹爪位置。
用于模型的 proprio 是原始state，匹配本体距离使用原资产 quantile 标准化state；
不额外拟合可能泄漏的全数据标准差。

`actions` 必须已经是训练基座使用的 LIBERO 7维 OSC 控制命令。
它不是简单 EEF 实际位移，也不是 CAP-X 8维关节目标。
本模块不会重做之前的关节→OSC回放转换，不新增动作，也不裁剪超限值。
旧转换时的额外补偿动作通过 source.action_mapping 追溯保留。
归一化后仅在模型维度补零，其有效mask为false。

## 初始场景与划分

所有父轨迹ID必须全局唯一，不能给每段换一个 episode_id。
相同初始场景的重复 rollout 必须同 initial_group_id。
优先用完整初始仿真状态哈希；若使用 suite/task/trial，需核实属于同一套初始化资源和索引。
自动manifest因此默认 initialization_verified=false，不声称仅靠trial号已证实相同初态。
默认 conservatively 按 task 整体分组；如设 conservative_task_split=false，
仍会对缺乏初始化证据的 task 整体合并。组件数量足够时保证非零的 train/validation/test
各有一个组件；不足时报告可能缺少某划分，不拆散父轨迹凑数。

## 标签和起点证据

四类标签直接复用已有切分结果。以下区别用于如实解释标签来源，不作为额外人工复核门槛：

- VLA 错误片段标签来自原 WCM 切分；本实验使用该负类标签，来源不声明为逐动作人工复核。
- “已在错误状态”不证明随后的补救动作也是错误动作。
- “该完整代码没有完成任务”不证明它没有推进任务。
- “最后一段代码最终完成任务”不证明这整段每个局部窗口都为正。
- 夹爪闭合不能单独证明 held_object；未知用 JSON null，明确空手用 "empty"。
- 已知不接触用 "none"；无法确认接触用 null，不能用 none 填空。
- phase/target/held/contact及critical_state_verified仅为可选历史元数据，不参与匹配，
  缺失或冲突均不影响匹配。不得为了匹配去补仿真真值／人工确认标志。

任务进度已从匹配条件中移除。progress 与 progress_scorer_id 仅作为可选历史记录保留，
缺失、进度差大或评分器版本不同均不影响覆盖、候选排序和正负配对；复核清单不要求补进度。
如额外记录进度，应是执行前观测的起点进度，统一[0,1]，并注明评分器版本，不作为动作质量。
**转换后code的起点图像发生变化，不得直接沿用原关节控制rollout的WCM分数**。
如需分析转换后code的进度，应另行评分；这不再是匹配前置条件。

匹配输入白名单：公开原任务指令、起点前视RGB、起点腕部RGB和模型实际输入state8。
公开指令相同才检索；不读取私有task_id、阶段、目标物体、持物／接触、物体位姿或未来帧。
state8前3维计算位置差，中3维旋转向量计算旋转差；末2维夹爪传感值参与归一化本体距离。
EEF侧字段仅留作日志，不作为独立真值来源。只有两路RGB特征进入图像距离。
已有四类标签仅负责正负池归属，episode／initial_group元数据仅负责防泄漏及参考来源追溯。
匹配结果和校准报告标注 observation_policy，拒绝复用旧特权筛选产生的prepared结果。

## 窗口与配对资产

`Window` 保存用户要求的身份、任务、事件、起点、图像/状态/EEF/夹爪/持物、进度、动作合约、
局部质量及证据。动作 NPZ 包含 `[H,D]` actions、valid_mask、trusted_error_mask。
H默认10；尾部补齐和D32中25个padding维度均无效。

稀缺负窗口逐步保留，其余约10控制步取样，额外保留事件起点。
交接起点另存一条 VLA unknown 的 state-only 记录供状态清单使用，不参与成功入池匹配。
各段共享父episodeID，采样不会把拆出的多个代码段当作多个独立失败episode加权。

pair mask 是正负有效mask∩本实验负类错误范围∩可比较维度，去掉共享动作前缀。
该错误范围默认取已有负类窗口的有效动作；有明确细粒度负标签时取其可信错误 mask。
不足3个有效动作步、空mask、近乎相同动作或同一动作起点标签矛盾的pair被排除。
找不到正参考的负窗口保留在训练池索引中，但采样器只从 pairs.jsonl 中选择负窗口。

prepared 中每个pair保存完整正负起点、程序与动作映射源、mask、匹配距离、门槛及冻结库ID。
训练前核查索引/配置/动作payload校验和，防止修改资产后继续旧run。
