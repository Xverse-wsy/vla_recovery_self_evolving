# Stage2：专家参考池与逐帧负样本匹配统计

本入口只建库和统计，不训练模型。Stage2的loss和正式训练预算仍待讨论。
用户授权本独立参考池使用专家40；Stage1仍保持专家只参加FM、CAP只查询VLA正库的规则。

**本版Stage2已选定：`configs/libero_pi05_stage2_numeric20.json`，
位置差2cm、旋转差10°、标准化state8 RMS0.20，前视/腕部图像距离各≤0.002。**
A800对应结果为`outputs/stage2_dense_expert_numeric20_20261101_v1/`：
144个可用负窗口，覆盖26条父轨迹，去重有效负动作205帧。
仅选择现有Stage2专用配置；Stage1代码、配置、数据及共享匹配实现均不修改。

## 本次做什么

1. 读取本机已经审计通过的Stage1密集全train正池：专家40、自主成功20、VLA正常推进片段、既定14个CAP成功后缀。
2. 建立独立正参考池，保留expert/VLA/code来源，不将专家伪装成VLA。
   不重新匹配CAP、不扩大成功后缀；当前正池仍为16324个观测起点。
3. 从恢复成功20＋失败20父轨迹中已经切分的40个VLA错误片段，枚举每一个真实观测起点。
   旧库39段各8帧、1段1帧；本轮共313个查询起点。所有起点都查询，包括尾部不足3步的窗口，
   但动作检查仍要求至少3步有效交集，因此短尾部可能匹配到状态却不能构成pair。
4. 每个负起点独立寻找正参考、计算mask；不继承上一帧的匹配，不按整段后缀扩张负样本。
   一个样本是一个匹配观测起点和其H=10动作窗口，窗口不跨标签/事件/控制权边界。

Stage1中成功末动作的重复补齐只用于FM；本轮pair参考的valid_mask恢复为真实动作mask，
重复动作不进入pair比较。负动作始终只保留真实动作，短尾无效位置为0且mask为false。
VLA中途正参考继续使用Stage1既定的完整真实H窗口；本入口不新增其中被Stage1舍弃的尾部起点。

## 匹配规则

只用公开任务指令、起点state8、前视/腕部RGB；没有任务进度、物体真值、接触真值或未来观测。
统一ActionSpec、归一化资产和字段顺序。本版Stage2使用用户选定的门槛：

| 项目 | 门槛 |
|---|---:|
| 起点末端位置差，从模型state8计算 | 2cm |
| 起点末端旋转差，从模型state8计算 | 10° |
| 标准化state8 RMS | 0.20 |
| 前视/腕部图像余弦距离，各自检查 | 0.002 |
| 数值候选上限 | 50 |
| 每个负窗口保留正参考上限 | 3，优先不同episode |

先同公开指令数值粗筛并排序，再用冻结的原始π0.5视觉编码器精筛。
动作mask为双方真实有效动作、可信错误范围、可比较7维的交集，移除共享近似动作前缀；
有效交集不足3步或动作几乎完全相同的候选排除。不平均正动作。
本轮不训练/验证/测试划分，记录均为train，同时保留原split作为来源信息。
这不会改变Stage1的匹配分组、准入结果或原文件。

## 可迁移执行命令

先按[README_STAGE1.md](README_STAGE1.md)在本机完成prepare；不要复制A800绝对路径索引。
保持同一训练环境、原始模型和数据，进入pi05_recovery_pair目录。
下例`PAIR_RUN`指向当前机器Stage1流水线输出，`STAGE2_MATCH_OUT`必须是新的目录。

```bash
source scripts/my.local.env
export STAGE2_MATCH_OUT=/data/pi05-runs/stage2_dense_expert_numeric20_v1
nvidia-smi
# GPU0仅是例子：选择有空闲显存的卡。8GiB allocator上限另需4GiB空闲余量。
mkdir -p "$(dirname "$STAGE2_MATCH_OUT")"
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PAIR_PYTHON" -u -m recovery_pair.stage2_matching \
  --positive-prepared "$PAIR_RUN/prepared" \
  --negative-windows "$PAIR_RUN/banks/windows/windows.jsonl" \
  --checkpoint "$PAIR_CHECKPOINT" --config configs/libero_pi05_stage2_numeric20.json \
  --output "$STAGE2_MATCH_OUT" --device cuda:0 --max-cuda-memory-gb 8 --batch-size 2 \
  > "$STAGE2_MATCH_OUT.log" 2>&1
```

可将上述命令放入screen运行；程序主动输出阶段进度，状态写入新目录status.json。
此处显式传Stage2专用配置；不要改Stage1的PAIR_CONFIG、algorithm.json或通用libero_pi05.json。
本入口不调用Stage1的FrozenMatcher来接收专家，避免修改Stage1的专家隔离规则。
完整回归还检查Stage1继续拒绝专家作为其匹配参考。

## 输出与计数口径

- `positive_references.jsonl`：独立的密集正参考索引，真实动作mask，不包含FM重复监督。
- `negative_queries.jsonl`：每帧负起点索引，动作与起点原观测直接对应。
- `negative_scopes.jsonl`：40个原负片段的边界、来源和哈希。
- `numerical_candidates.jsonl`：每个负起点最多50个候选和数值距离。
- `windows_with_features.jsonl`、`feature_cache/`：候选起点的冻结视觉特征。
- `pairs.jsonl`：逐起点配对、最多3条真实正参考、距离/门槛、mask和完整起点来源。
- `negative_window_results.jsonl`：每个起点的数值、图像和动作筛选结果，包括失败原因。
- `report.json`：最终统计，`training_started=false`、`formal_training_prepared=false`。

重要口径：

| 报告字段 | 含义 |
|---|---|
| `negative_query_windows` | 独立查询的负观测起点数，当前313 |
| `paired_negative_windows` | 可用负样本数：一个匹配观测＋一个动作窗口 |
| `paired_negative_segments` / `paired_negative_episodes` | 有至少一个可用窗口的原片段/父轨迹数，各计一次 |
| `selected_positive_references` | 配对关系数，多个参考不增加负样本数 |
| `raw_action_slots_in_paired_windows` | 配对窗口内真实动作位置数，重叠窗口会重复计数 |
| `unique_raw_action_frames_in_paired_windows` | 按父episode＋原动作步去重后的真实负动作帧 |
| `effective_action_slots_union_per_window` | 每个窗口先合并多个正参考的有效mask，再累加窗口；重叠窗口仍会重复 |
| `unique_effective_negative_action_frames` | 再按父episode＋原动作步去重，不将相邻窗口重叠动作算成新增帧 |
| `paired_windows_with_expert_reference` | 保留了至少一个专家正参考的负窗口数 |

参考数量/通过门槛不等于匹配准确率或模型成功率。旧结果15窗口来自40个稀疏起点及无专家参考池，
不能作为本轮313起点密集查询的结果，也不能直接归因所有数量变化都来自专家。
这份输出是匹配统计资产，不是已经接入Stage2优化器的正式训练prepared目录。

## 历史严格基线实际结果（2026-11-01）

输出：`outputs/stage2_dense_expert_matching_20261101_v1/`。

| 项目 | 数量 |
|---|---:|
| 正参考观测起点 | 16324：专家6632、自主成功3232、VLA正常推进4727、已准入CAP成功1733 |
| 独立负查询起点 | 313，来自40段/40条父轨迹 |
| 至少一个数值候选 | 107 |
| 至少一个图像通过候选 | 102 |
| 动作mask检查后可用负窗口 | 86，覆盖18段/18条父轨迹 |
| 所选正参考关系 | 199：VLA180、CAP3、专家16 |
| 至少选到一个专家参考的负窗口 | 6 |
| 窗口内有效动作位置累加 | 492，含相邻窗口重叠 |
| 按父轨迹及原动作步去重后的有效负动作帧 | 141 |

313个起点中，206个没有数值候选，5个没有图像通过候选，16个通过状态匹配但没有
足够有效动作交集，剩余86个可配对。39个原负片段各只有8个真实动作、另1个只有1个；
每段最后1～2帧仍独立查询，但不能凑出至少3步的pair，不会通过重复动作扩大样本数。
40条父轨迹存在负片段，并不保证每条负起点都能在当前门槛下找到相似正状态。

冻结视觉编码器在物理GPU0运行，峰值allocated约1.572GiB，原生图像预处理误差0。
额外逐一核对全部313个查询的原Parquet行号、起点state8和真实归一化动作，最大误差均为0。
本机113项测试通过，包含逐帧展开、专家独立参考、邻帧不继承、真实尾部mask、重叠帧去重
和合成端到端原资产不变检查。没有启动Stage2训练或LIBERO回放。

## 数值门槛放宽对照

用户随后要求试验稍宽的数值门槛。新增两个独立配置，不改上面的严格基线或Stage1配置：

| 配置 | 位置差 | 旋转差 | 标准化state8 RMS | 双视图距离各自上限 |
|---|---:|---:|---:|---:|
| `configs/libero_pi05.json`，严格基线 | 1cm | 5° | 0.10 | 0.002 |
| `configs/libero_pi05_stage2_numeric15.json` | 1.5cm | 7.5° | 0.15 | 0.002 |
| `configs/libero_pi05_stage2_numeric20.json`，本版选定 | 2cm | 10° | 0.20 | 0.002 |

仍逐负起点独立匹配，数值候选最多50个、正参考最多3个、至少3步有效动作交集，
不改变标签、真实动作mask和不补齐负尾部的规则。不启动优化器或回放。
沿用前文命令，将`--config`换为上述配置文件，并为`--output`指定另一个全新目录。
数值变宽只表示起点容差更大；计数增加不代表匹配准确率提高。

1.5倍数值门槛已在A800完成：162个起点有数值候选、148个通过图像，最终115个可用负窗口，
覆盖21段/21条父轨迹、去重有效负动作168帧。原严格基线的86个可用起点全部保留；
新增29个可用起点均来自原来206个无数值候选起点，其中55个现在至少有数值候选。
输出：`outputs/stage2_dense_expert_numeric15_20261101_v1/`。

2倍数值门槛也已完成：210个起点有数值候选、190个通过图像，最终144个可用负窗口，
覆盖26段/26条父轨迹、去重有效负动作205帧。
输出：`outputs/stage2_dense_expert_numeric20_20261101_v1/`。
两个实验都保持原输入哈希，不启动训练。用户已选定numeric20作为本版Stage2配置；
Stage1和通用严格配置仍保留原值，旧匹配结果不覆盖。
