# π0.5 local-window FM + continuous pair repulsion

这是用户 12 点方案的一版独立 PyTorch 实现。位于新目录，不修改已有 CAP-X、
RLinf 或旧 SFT 配方。实际验证范围见 [VALIDATION.md](VALIDATION.md)。

## 已实现的行为

| 环节 | 实现 |
| --- | --- |
| 数据 | 直接读取现有 selection.json 和片段边界；已有四类直接作为训练标签 |
| 防泄漏 | 完整 parent episode 与初始场景连通分组后划分，缺失初始化证据时按 task 保守分组 |
| 表示 | 原 checkpoint quantile 资产、7维 OSC 命令、32维填充；不做额外 delta 变换 |
| 窗口 | 控制权、程序段、标签改变、episode 边界截断；补齐无效；执行前图像和状态 |
| 成功筛选 | CAP 成功片段有窗口匹配冻结 VLA 成功库后，从最早匹配点纳入到同一片段结束 |
| 匹配 | 只用部署输入：公开任务指令、本体观测、前视／腕部RGB；数值筛选→最多50候选→图像比较 |
| 配对 | 每负窗口≤3条真实正参考，优先不同 episode；保存 mask、距离、门槛、起点及来源 |
| FM | 正样本自己的起点及真实动作；保留原 Beta(1.5,1)×.999+.001 时间采样 |
| pair | 负样本起点、纯新噪声、完整 K 步可微 Euler；无 hinge，无 detach |
| 采样 | episode→event→负窗口→正参考；每更新每负 episode 一票；最多50次复用 |
| 训练 | FM64 / pair≤4 分别平均；microbatch、checkpointing、多卡 SUM、断点及验证提前停止 |
| 对照 | continuous_pair / pull_only / fm_only，相同数据及逻辑 batch 规则 |

`loss = mean(FM) + eta * mean(pull - lambda_push * push)`。
pull/push 在归一化空间使用完全相同的 mask 和距离。mask 排除 padding、共享动作前缀、
不在所选负标签范围的元素和不足的窗口。负 pair loss 合法；要求 `0 < lambda_push < 1`。

当前 LIBERO 配置为 **H=10、D=32、环境动作7维、K=3 Euler、20Hz**。
H 与 K、部署时实际执行的 action chunk 长度是不同参数。原数据 `fps=10` 是索引约定，
此处不会据此重采样到10Hz，也不再次旋转已导出的 RGB。

匹配已移除任务进度条件：覆盖筛选和正负配对均不检查进度差、进度是否缺失或评分器版本，
候选排序也不使用进度。历史进度字段仅作可选记录。旧配置中的 progress 门槛读取时忽略，
新配置不再导出该门槛。当前匹配禁止特权信息：不读取阶段、目标物体标签、持物／接触真值、
仿真物体位姿、人工确认标志、单独的EEF日志或未来观测。公开任务指令相同后，
从模型实际输入的state8计算末端位置／旋转差和归一化本体距离，再比较两路RGB特征。
旧object_position_m门槛读取时忽略。task_id仅用于数据分组／追溯，不用于相似性筛选。

## 安装与快速验证

CPU算法验证可以单独安装；不需要仿真、API 或模型权重：

```bash
cd /path/to/cap-x/pi05_recovery_pair
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python -m recovery_pair smoke --output outputs/my_smoke --updates 2
```

若使用 uv，请加 `uv --no-config pip install ...`，避免继承上层 CAP-X 的依赖覆盖。
Python3.13 的新版 PyArrow 需要 NumPy2；本地验证版本见 VALIDATION。

真实模型须使用已有 π0.5 训练环境：PyTorch2.7、CUDA、RLinf/OpenPI 的依赖和 tokenizer
资产均需已安装。**本目录的 pyproject 只提供新增算法依赖，不是整套 π0.5 环境安装器**。
可复用独立 SFT 仓库的训练环境：
https://github.com/Xverse-wsy/vla_recovery_self_evolving 。

后端绑定 RLinf 提交 `c70606f08cdca259b8dec03d4430926b5b8fac9d`，实际加载时检查版本与源文件。
支持 editable checkout，也支持 pip 安装的源码副本；副本需通过 direct_url.json 找到原始
Git checkout，再逐个比对 OpenPI 模块和 config.py，拒绝版本或代码内容不符的环境。
不接受把另一版 OpenPI 的接口当作同一模型。必要时另建 checkout，不切换他人正在用的仓库。
在已有 RLinf 环境中安装新增模块：

```bash
python -m pip install --no-deps -e /path/to/pi05_recovery_pair
python -c 'import torch, numpy, pyarrow, PIL, omegaconf, rlinf, openpi'
```

## 真实数据准备

建库已包含在代码中，A800 只用来验证。同一份代码可以在训练服务器上直接读取已有
切分数据重建索引，不需要复制 A800 的输出。没有写死服务器 IP、用户目录或 GPU 号。

先改 `configs/data_sources.example.json` 的数据路径；相对数据路径均相对于 `--data-root`，
`algorithm_config` 则相对于这个 JSON 文件所在目录。数据目录结构保持一致时只改 data-root。
数据条数变化时同步修改／移除 `expected_segment_counts`，它只用于核对当前数据清单。

从本目录执行一条命令完成四类片段建库、父轨迹划分、模型训练窗口索引与完整性检查：

```bash
python -m recovery_pair build-libraries \
  --sources-config configs/data_sources.example.json \
  --data-root /path/on/training/server \
  --output /path/to/new_output
```

不安装本模块时，也可用绝对路径运行 `python /path/to/pi05_recovery_pair/scripts/build_existing_libraries.py`
并传相同三个参数；不依赖当前工作目录。此步骤仅用 CPU，无训练、仿真或 API 请求。
每次使用新输出目录，已有目录会拒绝覆盖。`status.json` 和 `report.json` 记录状态及源文件哈希。
更换机器后重新运行此命令，因为生成的窗口索引包含该机器的绝对数据路径。
需要补过人工复核的 manifest 时，可在来源配置的 `manifests` 列表中指定，不要与同一批
segmented/autonomous 来源重复导入。`--data-root` 下需保留完整数据、meta、provenance和
selection.json；不能仅有视频或原40条／自主20条／未切分恢复20条。

以下是分步命令，便于单独检查各阶段。环境变量里的路径均应改为实际位置。

```bash
ROOT=/home/wenchaoxu/wsy/vla-recovery-posttrain-loop
NORM="$ROOT/models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json"
python -m recovery_pair default-config --output config.json
python -m recovery_pair manifest \
  --segmented-dataset "$ROOT/dataset_subsets/pi05_recovery_segmented/recovery_success20_segmented" \
  --segmented-dataset "$ROOT/dataset_subsets/pi05_recovery_segmented/recovery_failure20_segmented" \
  --autonomous-dataset "$ROOT/dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot" \
  --output manifest.jsonl
```

`manifest` 只读现有文件，保留父轨迹、程序、动作映射，并从原 `selection.json`
直接映射 `bank_class`。建库不依赖视觉编码器、门槛校准或配对，不重新切分原 episode。

| 原有分类 | 四类库 | 当前片段数 |
| --- | --- | ---: |
| vla_progress_candidate ＋自主成功20条 | D_vla_positive | 60 |
| vla_wcm_warning | D_vla_negative | 40 |
| capx_completed | D_code_positive | 20 |
| capx_not_completed | D_code_negative | 362 |

按用户最新实验定义，以上四类直接作为训练标签：VLA 正常／自主成功、VLA 到达错误状态、
CAP 恢复成功、CAP 恢复失败。`training_quality` 从 `bank_class` 读取正负，
不再因旧 `local_quality=unknown` 而排除。来源仍如实记为原 WCM／程序完成分类，
不伪造逐动作复核标志。有单独的细粒度标签时保留其证据，矛盾标签报错。
按 [DATA_CONTRACT.md](DATA_CONTRACT.md) 使用已有起点图像与state8；不要求补持物／接触等标注。

自主20条通过 `--autonomous-dataset` 读取，`controller=VLA`，一条完整原 episode 保持同一ID。
自主成功的有效窗口通过原分类作为正标签。两个入口均使用 canonical_task_index
作为任务编号，避免将旧日志的评测列表编号与 LIBERO 官方任务编号混淆。
原始专家40不伪装成 VLA 状态；如要额外加入专家 FM，
需明确增加 expert 角色与采样规则，本版没有悄悄加入。

```bash
python -m recovery_pair libraries --manifest manifest.jsonl \
  --config config.json --output outputs/libraries_v1
python -m recovery_pair audit-libraries --libraries outputs/libraries_v1
python -m recovery_pair build --libraries outputs/libraries_v1 --norm-stats "$NORM" \
  --config config.json --output outputs/windows_v1
```

`libraries_v1/segments/` 保存四类现有片段索引，`splits/` 保存先按完整父轨迹／初始场景
分好的三个划分；此时尚未切训练窗口。`windows_v1/banks/` 保存四类窗口索引及各划分的
B_positive、B_negative、S_vla_all。交接 state-only 观测只加入状态库，不计入四类片段。
另输出 `windows.jsonl`、独立动作 NPZ、`review_queue.jsonl` 及输入 manifest。
数据源不移动、不改写。修改标签或边界后另建输出，不原地修改训练资产。
也可用 `build --manifest ...`，它同样先分组划分，再建立窗口。

## 冻结视觉特征与门槛校准

如需先统计数值粗筛数量、暂不做图像比较，可运行：

```bash
python -m recovery_pair numeric-preview \
  --windows outputs/windows_v1/windows.jsonl --config configs/libero_pi05.json \
  --output outputs/numeric_preview_v1
```

该命令查询CAP成功窗口→VLA成功窗口，保留相同公开指令与split约束，输出窗口／原片段／
episode计数、最多50个数值候选及距离。若proprio_rms尚为null，只应用位置／旋转门槛，
另报告RMS分布，明确标记not_applied_uncalibrated；可用`--proprio-rms VALUE`显式检查指定门槛。
这里不读图像、不生成可靠视觉配对、不改变训练池。结果见新目录report.json与numeric_candidates.jsonl。

提供已选好的固定 TorchScript 视觉编码器，输入 `[B,3,H,W]`，输出 `[B,F]`。
本实现不自动下载模型，也不把随机网络当作可靠编码器。
预处理示例文件见 `configs/visual_preprocessing.example.json`，须与编码器的训练预处理一致。
只编码前视和腕部RGB两路部署图像；额外的分割图、深度真值或仿真视角不会读入。
不以物体真值位姿、接触或持物标签作为匹配条件。

### 复用 π0.5 原生视觉模块的小规模试验

也可直接复用已有 RLinf π0.5 checkpoint 的视觉权重，无需导出 TorchScript：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m recovery_pair.vision_trial \
  --windows outputs/windows_v1/windows.jsonl \
  --numeric-preview outputs/numeric_preview_v1 \
  --checkpoint /path/to/RLinf-Pi05-LIBERO-SFT \
  --output outputs/vision_rms_trial_v1 \
  --device cuda:0 --max-cuda-memory-gb 8 --batch-size 2 --review-count 36
```

先检查所选物理卡空闲显存；这里的0可按机器修改。CUDA allocator上限8GiB，启动要求
至少额外留4GiB空闲。只加载原模型的 SigLIP 和视觉投影权重，strict加载并冻结，
复用上游权重转换、ResizeImages和输入范围，核对原生预处理；不加载语言或action expert。
用原始起点前视／腕部RGB分别提取投影patch token的均值并做L2归一化，不翻转图像。
本命令仅编码数值预览中涉及的候选窗口，不做训练、不修改正式正池或原数据。

输出 `report.json`、`joint_distances.jsonl`、`threshold_grid.json`、图像特征缓存、
带特征的候选窗口索引，以及按图像距离排序分层抽样的起点对照图和复核模板。
RMS与图像距离的联合分析是同时检查两个距离，不是把它们相加；图像门槛使用两路距离
的最大值，两路必须同时满足。门槛网格中的数量只是候选数量，不是可靠配对或准确率。

复核时只看执行前RGB和模型本体观测，填写模板的 `compatible`（true/false/null）、
`reviewer`、`evidence`，另存JSONL，不能用未来结果或仿真真值。可重复运行审计：

```bash
python -m recovery_pair.review_trial \
  --trial outputs/vision_rms_trial_v1 \
  --reviews /path/to/reviewed_pairs.jsonl \
  --output outputs/vision_review_audit_v1
```

审计核对pair身份、只接受train候选、unknown不算负例，保存源文件哈希。
兼容或不兼容复核样本不足5个时明确报告证据不足；即便数量足够，也不会自动启用门槛。
分层小样本有episode相关性，不能据此报告总体准确率。2026-10-30的真实试验发现全图
平均特征仍会把物体配置不同的状态判为接近。用户随后明确接受这种差异，因此复核中的
物体配置冲突不作为额外排除条件。当前采用下述初始门槛，尚未验证匹配准确率。

```bash
python -m recovery_pair features --windows outputs/windows_v1/windows.jsonl \
  --encoder /path/to/frozen_encoder.pt --preprocessing configs/visual_preprocessing.example.json \
  --cache outputs/feature_cache --output outputs/windows_with_features.jsonl --device cpu
```

通用 `Config()` 的本体与图像门槛默认为 null，未设置时匹配会拒绝运行。
当前 `configs/libero_pi05.json` 已设置本轮初始门槛：

| 条件 | CAP成功入池 coverage | 负窗口找正参考 pairing |
| --- | ---: | ---: |
| 末端位置差 | 0.05m | 0.01m |
| 末端旋转差 | 30° | 5° |
| 原资产归一化state8 RMS | 0.50 | 0.10 |
| 两路图像余弦距离的最大值 | 0.007 | 0.002 |

该门槛绑定本轮的原生π0.5视觉权重、均值池化和归一化资产；换编码器或特征处理后需重新
检查数值尺度。选择依据是本轮train候选的距离分布，以及用户允许物体配置差异的标准。
最新用户要求放宽入池并采用成功片段后缀：`cap_success_admission=segment_suffix`。
同一成功代码片段中有任意窗口匹配，就从最早匹配窗口的起点，纳入到该片段结束的连续
后续动作；前缀、其他程序段、失败程序不纳入。后续窗口不需要各自再匹配一次。
它们按成功代码片段来源纳入，不能声称每个后续起点都已和VLA匹配。源动作和mask不变，
只是正池索引扩展；保留原片段边界以及每个窗口自己的真实观测和动作。

当前放宽门槛下，实际数值候选641组，视觉通过501组关系、115个直接匹配的CAP窗口；
按后缀规则纳入14/20个原成功片段的后续部分，共1733个不重复原动作步、182个训练窗口。
此前0.20／0.005下的27窗口／6片段结果是历史设置，保存在v1输出中。
pairing使用更严格的初始值，尚未在全部负窗口上统计配对数量。
上述数量是覆盖统计，没有匹配准确率结论。原四类标签与匹配顺序保持不变，物体位置差异、
人工画面复核标签不额外否决候选，不读取物体真值。已有试验与审计输出保留不变。

如果后续需要基于新的兼容性复核进一步校准，可以使用以下独立入口：
准备仅来自 train 划分的候选状态兼容性复核：

```json
{"query_id":"...","reference_id":"...","compatible":true,"reviewer":"..."}
```

```bash
python -m recovery_pair calibrate --windows outputs/windows_with_features.jsonl \
  --reviews reviewed_state_pairs.jsonl --config config.json --output outputs/calibration_v1
```

校准提供正兼容样本95分位初始候选门槛及误接纳报告，**不能据此直接宣称识别准确率**。
检查报告；如有误接纳，根据公开图像和本体观测收紧门槛。校准复核也只看这些部署输入，
不得使用仿真隐藏状态或未来结果。pair门槛不得宽于coverage。
最终设置由 `outputs/calibration_v1/config.json` 保存。

```bash
python -m recovery_pair prepare --windows outputs/windows_with_features.jsonl \
  --config outputs/calibration_v1/config.json --output outputs/prepared_v1
python -m recovery_pair audit --prepared outputs/prepared_v1
```

匹配顺序固定：①原 VLA 正片段作为冻结参考；②CAP 成功窗口按起点匹配该库，从每个成功
代码片段最早匹配点纳入到该原片段结束；
③VLA 错误和 CAP 失败窗口在正池中寻找严格正参考。接受的 CAP 不能反过来放行别的 CAP。
失败窗口不执行第一步的成功入池匹配；未找到可靠正参考的负窗口保留，但不做负训练。

输出所有窗口、VLA 状态清单、`frozen_vla_success.jsonl`、成功匹配与最终配对、统计和校验和。
`success_matches.jsonl` 保存每条 CAP→VLA 成功匹配的起点、距离、门槛、标签来源；
`admitted_cap_success_windows.jsonl` 是纳入的全部CAP窗口（包括无需单独匹配的后续窗口）；
`directly_matched_cap_success_windows.jsonl` 仅含自身匹配到VLA的窗口。
`matched_cap_success_windows.jsonl` 同样仅保存直接匹配窗口，数量与report的matched计数一致。
`cap_success_suffixes.jsonl` 保存每段的锚点、VLA参考、原Parquet行范围和父轨迹步数范围，
均为零起点右端不包含。`coverage.jsonl` 区分directly_matched和suffix_anchor。
`eligible_banks/{train,validation,test}/` 保存筛选后的可用训练索引，
与前面 `windows_v1/banks/` 保存的原有分类库分开。
没有四类归属、也没有显式标签的 unknown 动作不进入 FM/pair；handoff 仅作状态记录。
匹配只在各自划分内部进行。
输出带 `observation_policy=policy_inputs_only_v1`；旧特权条件产生的 prepared 匹配结果
不能直接用于当前训练，须用现有窗口重新提特征／prepare，不需要改原数据。

### 单独运行CAP成功后缀入池

此阶段可独立验证，不启动负配对或训练。下面的脚本包含数值候选、原生冻结视觉特征和
成功后缀入池三步；需已有π0.5模型环境，先检查显存，GPU由CUDA_VISIBLE_DEVICES选择：

```bash
CUDA_VISIBLE_DEVICES=0 PAIR_PYTHON=/path/to/pi05-env/bin/python \
bash scripts/run_success_admission.sh \
  /path/to/windows.jsonl /path/to/RLinf-Pi05-LIBERO-SFT \
  configs/libero_pi05.json /path/to/new_output cuda:0 8 2
```

脚本不依赖当前工作目录来导入模块，参数中的相对文件路径按启动目录解析。
配置在启动时快照保存；输出不可覆盖，run_status.json记录阶段，异常时记录failed。
GPU allocator上限8GiB、batch2，启动检查额外4GiB空闲空间。每次用新输出目录。
如果已有同源窗口的完整候选视觉特征，可仅运行CPU入池阶段：

```bash
python -m recovery_pair.success_admission \
  --windows /path/to/windows.jsonl --feature-trial /path/to/vision_trial \
  --config configs/libero_pi05.json --output /path/to/new_admission
```

此入口核对原窗口哈希、特征记录与源字段、全部数值候选特征完整性，并检查后缀连续覆盖
至原Parquet程序段末尾。输出positive_pool.jsonl、admitted_cap_success_windows.jsonl、
success_matches.jsonl（直接匹配）、cap_success_suffixes.jsonl、coverage.jsonl和report.json。
它们是可直接追溯原动作payload的训练窗口索引，不额外复制或转换原数据。
尚未为全部负窗口和新正参考编码，不能把此成功入池输出作为完整prepared目录启动pair训练。

## 训练、对照、多卡、恢复

先查看 `nvidia-smi` 和他人进程，手工选择空闲设备；脚本不 kill 进程、不自动抢卡。
每卡默认 microbatch1，可改配置中的 fm_micro_batch / pair_micro_batch；
全局逻辑 batch 与 world size 分开，不随 GPU 数量自动扩大。

可先独立进行真实模型审计，允许使用未标注的真实起点，不做优化器更新：

```bash
CUDA_VISIBLE_DEVICES=0 python -m recovery_pair audit-model \
  --windows outputs/windows_v1/windows.jsonl --config configs/libero_pi05.json \
  --checkpoint "$ROOT/models/RLinf-Pi05-LIBERO-SFT" --norm-stats "$NORM" \
  --output outputs/model_audit.json --device cuda:0 --max-cuda-memory-gb 14
```

GPU审计默认保留至少4GiB空闲余量；14GiB是PyTorch allocator上限，CUDA上下文本身还会
占少量显存。审计检查真实输入归一化、原生/可微采样前向和全部K步梯度，分别测试checkpoint
开关。它不建立局部行为标签，也不代表pair训练或LIBERO成功率已验证。

```bash
BASE="$ROOT/models/RLinf-Pi05-LIBERO-SFT"
CUDA_VISIBLE_DEVICES=YOUR_FREE_GPU python -m recovery_pair train \
  --prepared outputs/prepared_v1 --config outputs/calibration_v1/config.json \
  --checkpoint "$BASE" --norm-stats "$NORM" --output outputs/continuous_v1 \
  --max-updates 1
```

此命令会加载真实模型并做本机短验证／更新。执行前确认设备空间。
首次更新前自动比对原生 sampler 与可微 sampler 的相同噪声前向及逐步梯度。
模型只训练 action expert 及必要投影，冻结视觉/语言前缀；FM 保留原时间采样，
默认关闭图像随机增强以与部署前缀一致。没有额外正向 warm-up。
这项图像增强选择是相对原始 SFT 的显式差异。

多卡使用 torchrun；不要再外套 DDP（这里已做全局分支均值和梯度 SUM）：

```bash
CUDA_VISIBLE_DEVICES=YOUR_FREE_GPU_LIST torchrun --standalone --nproc_per_node=NUM_GPUS \
  -m recovery_pair train \
  --prepared outputs/prepared_v1 --config outputs/calibration_v1/config.json \
  --checkpoint "$BASE" --norm-stats "$NORM" --output outputs/continuous_full
```

默认 lr1e-5、clip1、FM64、pair≤4、lambda_push=.1、eta=.05前10%升高。
每个有可靠负配对的独立 episode 最多50次被选中；并非把正样本池重复50遍。
无 pair 时使用显式 `fm_only_updates` 上限，仅训 FM。每50更新保存，默认保留最近3份。
若有验证划分正数据，则每25更新用固定验证窗口/噪声检查，连续5次无改善停止。
这是 heldout 训练目标早停，LIBERO实际成功率仍需独立仿真评测。

`--mode pull_only` 和 `--mode fm_only` 使用独立输出目录做两项消融。
同样可以用冻结 base checkpoint 做第三个部署对照。
用 `--resume outputs/continuous_v1/update_000001.pt` 在原输出目录显式恢复，
同时保留相同 base/data/config/mode；允许改 microbatch 和GPU数量。
checkpoint 含可训练参数、优化器、RNG和每个 episode 的复用计数；冻结参数来自原始 base。

训练检查点是增量文件，不直接伪装成完整基座。
可在有足够 CPU 内存和磁盘时显式物化为原 RLinf 可读 full_weights.pt：

```bash
python -m recovery_pair export --checkpoint outputs/continuous_full/update_000100.pt \
  --output outputs/exported_continuous
```

导出仍需原始 tokenizer、归一化资产和相同 H/K 配置。真实 LIBERO 整体评测继续使用独立
评测仓库的脚本；本目录不自动启动400/2000条仿真。

## 文件入口

- `libraries.py / data.py / schema.py / splits.py`：原分类建库、数据合约、切窗、父轨迹划分。
- `features.py / matching.py / pipeline.py`：冻结编码、双门槛、索引与可追溯配对。
- `flow.py / losses.py`：完整可微采样和统一 mask 损失。
- `batches.py / engine.py / trainer.py`：episode权重、逻辑batch、多卡与断点。
- `openpi_backend.py`：固定 RLinf π0.5 接口，不改上游源码。
- `tests/`：数值、数据、微批次、分布式和端到端验收。
