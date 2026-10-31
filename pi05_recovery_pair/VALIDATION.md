# 验证记录：2026-10-30

## 实际完成

在本机 macOS arm64 的独立 `.venv` 中执行：

```bash
python -m pytest
# 55 passed
python -m recovery_pair smoke --output outputs/smoke_cpu_20261030 --updates 2
# synthetic_only=true; 2 updates; validation enabled; checkpoints written
```

环境：Python3.13.5、PyTorch2.7.1、NumPy2.2.6、PyArrow26.0.0、Pillow13.0.0、pytest9.2.0。
测试中的本地双进程 Gloo 需要 IPC/localhost 权限。没有使用 GPU、仿真或付费API。

55项测试覆盖：

- H/K独立；K=1/3/7时可微Euler与独立部署参考前向完全一致，含checkpoint开关。
- 每个flow中间动作有梯度，冻结前缀无梯度；double精度有限差分与自动微分一致。
- sampler仅接受起点、纯噪声及K，记录正负动作不作为初始化。
- 统一masked距离对填充、非有限无效元素不敏感；空pair batch安全。
- FM输入中的无效记录动作先置零，不能通过action-token attention影响有效损失。
- 大距离仍持续排斥；负pair loss合法；lambda≥1等无效值被拒绝。
- 微批次累积与完整逻辑batch的梯度一致。
- 两个CPU rank的SUM梯度与单进程全局均值一致，包括一个rank没有pair的情况。
- CAP成功只匹配冻结VLA成功库；错误／unknown VLA和已接受code不能用于放行。
- 最多50视觉候选、最多3正参考、优先不同episode；只使用任务指令、RGB和模型本体观测。
- 同初态及完整episode不跨划分，初始化证据不足时保守合并整个task。
- 真实Parquet格式的合成数据读入、尾部mask、局部标签边界、交接state-only记录。
- 多视角TorchScript编码/缓存、图像方向、原生EnvIO字段及仅起点输入。
- quantile归一化不裁剪；校准不能读取validation/test。
- episode预算、无code回填、空pair/FM-only、训练断点恢复和固定验证窗口。
- 安装版RLinf源码副本与固定Git版本逐文件一致性；修改过的副本被拒绝。
- 自主数据按canonical任务编号导入，以已有四类作为训练标签，保留原来源证据。
- 原 selection 分类直接建四类库，无视觉门槛也可建库；负类有效动作作为实验错误范围。
- 先划分父episode再切窗；handoff仅进入状态库；不覆盖旧输出、拒绝分类冲突。
- 通用配置入口从任意工作目录运行，数据根目录与代码目录分离；条数错误明确中止。
- 删除进度匹配后，进度缺失／相差大／评分器版本不同均可通过覆盖和配对；
  输出距离、候选排序和新门槛配置不含进度，旧配置中的进度门槛被忽略。
- 原四类标签在local_quality保持unknown时可建立成功匹配、严格pair并通过审计；
  合成CPU完成一次参数更新。矛盾的原分类／显式局部标签被拒绝。
- 禁用全部特权匹配字段；任意冲突的phase/target/held/contact/物体位姿/私有task_id及
  关键状态确认标志不影响成功入池或正负参考；numeric接口只能访问公开输入白名单。
- EEF位置／旋转差从模型raw state8计算；额外仿真图像／特征不会进入图像比较。
- 旧prepared缺少公开输入策略版本时被拒绝，避免隐式沿用旧特权规则。

合成端到端样例使用6个虚构任务，四种owner/quality组合。它产生120个可靠合成负窗口配对，
训练划分8个独立负episode；两次更新均FM64/pair4。
sample audit最大前向差0，3个积分步骤均有梯度。
这些数字仅用于检查管线，不是LIBERO识别准确率或真实模型性能。

## SSH恢复后完成的A800验证

用户确认连接恢复后，已同步至新目录：

`/home/wenchaoxu/wsy/vla-recovery-posttrain-loop/training_configs/pi05_recovery_pair/`

没有改写原始数据、已有SFT脚本或已安装环境。CPU导入实际复用了：

- 20条CAP恢复成功父轨迹：177个已有段，43535动作。
- 20条CAP恢复失败父轨迹：285个已有段，82518动作。
- 20条VLA自主成功轨迹：3232动作。

177段并不表示177次恢复成功。它们是20段VLA报警前候选、20段WCM警戒片段和137段CAP
程序。仅20个CAP程序段包含最终完成任务动作，另117个没有完成；这仍不是局部行为正负标签。

首轮 `real_import_v1` 导入共482个已有段/整轨迹条目、60条完整父episode、24个canonical任务，
构建13164窗口（其中40个额外交接state-only记录）。所有局部质量保持unknown。
划分计数：train VLA699/code7586；validation VLA174/code3092；test VLA68/code1545。
起点图像读取、8维状态、7维动作、父轨迹分组、norm资产及真实任务指令映射均通过检查。
旧日志task_id是评测列表位置，本版改用canonical_task_index，避免任务错配。

norm SHA256：`dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`。
服务器pip安装版RLinf的59个OpenPI/config源码文件与固定提交逐文件相同。

真实π0.5检查按用户指定的**物理GPU0**运行，PyTorch allocator上限14GiB：

- 基座 RLinf-Pi05-LIBERO-SFT，H10、D32、K3 Euler。
- 窗口动作归一化与模型原始输入pipeline最大误差：2.3866e-7。
- 相同真实起点、噪声、K和网格：可微/原生sampler最大前向差0。
- checkpoint_sampling关闭/开启两种模式均通过，每种模式的3个积分步骤均有梯度。
- 冻结2923335408参数，可训练430098464参数（expert与必要投影）。
- 峰值allocated 7.6508GiB，reserved 7.7520GiB；检查正常退出。
- **没有优化器更新，没有启动真实长训、仿真或API请求**。

服务器证据文件：

- `outputs/real_import_v1/report.json`
- `outputs/real_model_gpu0_audit_v1.json`

## 直接复用现有分类的建库修正

此前导入保留了来源分类字段，却没有将其连接到四类库索引。已增加 `bank_class` 与
`bank_label`，从原 selection.json 直接映射并独立建库；局部训练可信度仍单独记录。
不重新切分原轨迹，不改原分类，不把“未完成整个任务”当成每个动作都错。

2026-10-30 在 A800 的新输出 `outputs/existing_banks_20261030_v2/` 完成 CPU 验证：

| 四类库 | 原有片段／整轨迹条目 |
| --- | ---: |
| D_vla_positive | 60 |
| D_vla_negative | 40 |
| D_code_positive | 20 |
| D_code_negative | 362 |

总计482条目、60父episode、24任务，训练窗口13164个，其中40个为额外交接状态记录。
四类库在视觉特征和门槛校准之前已经独立存在；源 metadata 哈希核对通过。
`libraries/segments/` 是原分段索引，`windows/banks/` 是四类窗口和 VLA 状态库。
完成建库不代表局部动作已复核、可靠pairs已匹配或训练已启动。

通用入口 `build-libraries --sources-config ... --data-root ... --output ...`
及 `scripts/build_existing_libraries.py` 已纳入代码。
来源配置不写死服务器路径；支持新机器直接读取原切分数据并重新生成绝对路径索引。
该入口已在 A800 从 `/tmp` 启动实际跑通，输出在
`outputs/existing_banks_portable_20261030_v3/`，相同482条目、60父episode、24任务和
13164窗口，四类条数一致，源 metadata 哈希未变。全程CPU，无优化器更新。
状态与统计见该目录 `status.json`、`report.json`；它验证的是同一台A800上的可迁移入口，
尚未在目标H100服务器运行。

## 当前用户指定的标签与匹配策略

用户随后要求直接使用已有四类标签，并先将CAP成功片段匹配VLA成功库。该策略已接入
matcher、配对mask、资产审计、采样器和训练器，不再由local_quality=unknown阻断。
原local证据与数据payload保留不变，pair记录明确标注使用原片段类别作为错误范围来源。
CAP失败与VLA错误片段留作后续严格负查询；它们不负责放行CAP成功片段。

A800只读检查原v3窗口：VLA正861、VLA负40、CAP正427、CAP负11796、交接状态40。
这些是窗口数量；CAP正来自20个已切分成功程序段。当前匹配参考池只有861个VLA正窗口，
查询池为427个CAP正窗口。起点关键状态标注要求现已移除；真实视觉编码器与门槛尚待配置。
55项CPU测试通过，包含四类原标签驱动的完整合成匹配／审计／一次参数更新；无真实训练。

用户最新要求禁止匹配使用特权信息，已同时修改成功入池、负配对索引、训练一致性检查和
校准路径。公开指令相同后，只比较模型输入state8与前视／腕部RGB特征，
保留3cm/15°成功入池和1cm/5°严格配对门槛，位姿来自state8而非独立仿真侧字段。
起点phase/目标物体/持物/接触/物体位姿/人工确认标志不再是运行前置条件。
已有四类标签仍负责池归属，数据分组元数据仍用于防泄漏，均不用于状态相似度。

## 尚未完成的真实验证

- 已有四类标签可直接用于匹配／训练，无需重标；不再需要起点关键状态特权标注。
- 已有原生视觉编码器的小规模试验（见下文），正式门槛校准、可靠pair数量和准确率统计未完成。
- 未进行真实pair/FM优化器更新、CUDA多卡训练、增量导出后LIBERO成功率评测。
- 真实模型未做有限差分检查；数值有限差分、多卡权重及完整训练器验收目前限于CPU合成测试。

## 真实库数值粗筛预览

新增可迁移命令 `numeric-preview`；其2项针对性测试通过，检查无图像／特权字段可运行、
未校准RMS显式跳过、可选RMS门槛、同公开指令及split限制、候选数上限与不改原输入。
这是55项基线测试后的新增测试，未重新执行整套57项。

A800实际统计输出 `outputs/numeric_match_20261030_v1/`：

- 查询427个CAP成功窗口，参考861个VLA成功窗口。
- 只使用3cm位置门槛和15°旋转门槛，RMS尚未校准，明确不启用；不读任何图像。
- 43/427窗口通过（10.0703%），涉及8/20原成功片段、3/11公开任务指令。
- 共29882个同指令同split比较，133组数值对应关系通过；未达到单窗口50候选上限。
- train为43/308，validation为0/74，test为0/45。
- 输出RMS分布与逐窗口候选；源窗口索引哈希未变，无GPU、无训练、未改正式匹配池。

这些不是完整视觉匹配或可靠训练pair的数量。正式匹配仍需视觉特征和RMS／图像门槛。
报告在 `report.json`，逐窗口候选与距离在 `numeric_candidates.jsonl`；CLI可在新机器同样运行。

原始专家40不是VLA rollout；目前未加入VLA覆盖库，也未静默扩展FM采样角色。

## 原生 π0.5 视觉与 RMS 小规模试验

2026-10-30，新增 `pi05_vision.py`、`vision_trial.py`、`review_trial.py`，
实际在A800物理GPU0的独立screen `pi05_vision_rms_trial`中完成，无优化器更新。
原数值预览的133组候选涉及99个起点窗口（43个CAP、56个VLA），编码198张前视／腕部RGB。
查询仍只对原冻结VLA正库，数值门槛3cm／15°，同公开指令同split，不读特权信息。

模型权重 `RLinf-Pi05-LIBERO-SFT/model.safetensors` 中只加载原生SigLIP视觉塔及投影，
复用RLinf的 `old_to_new_state_dict`，所有权重strict加载；冻结414803696参数，训练参数0。
编码采用投影后的patch token均值、L2归一化。原生ResizeImages→[-1,1]预处理对照误差0；
编码器encoder/head使用bf16，stem/position保留fp32；未旋转、未做随机增强。
GPU allocator上限8GiB，峰值allocated **1.57163GiB**；进程正常完成退出。
RLinf59个文件来源校验通过，视觉权重指纹为
`15f4b4b9f3c3208977702fe96c9c6fa671d99244c1ed56e9f9a20f00bad28bd8`。

候选距离分布（尚无准确率意义）：

| 距离 | 中位数 | 95分位 |
| --- | ---: | ---: |
| 标准化state8 RMS | 0.111996 | 1.001199 |
| 前视余弦距离 | 0.0003605 | 0.0005738 |
| 腕部余弦距离 | 0.0037048 | 0.0065468 |

本批候选的两路最大距离均来自腕部。门槛扫描示例，两路图像均须通过：

| RMS门槛 | 图像门槛 | 候选关系数 | CAP窗口数 | CAP原成功段数 |
| --- | --- | ---: | ---: | ---: |
| 不启用 | 不启用 | 133 | 43 | 8 |
| 0.20 | 0.005 | 70 | 27 | 6 |
| 0.20 | 0.002 | 29 | 9 | 3 |
| 0.10 | 0.002 | 28 | 8 | 3 |
| 0.10 | 0.001 | 5 | 3 | 1 |

这不是正式入池数量或配对准确率。36组按图像距离秩分层抽样的起点对照图已经查看；
仅依据起点两路RGB和原始state8做保守兼容性复核：30组有可见配置冲突，6组遮挡／关系
不明确，0组可明确确认兼容，97组未逐一复核。没有把不确定样本强行标负。
它们不是新增动作质量标签，也不是动作可迁移性的真值，复核判断不会改四类训练标签。

重要反例：复核#1的RMS约0.0445，前视约0.000300、腕部约0.000681，仍可见VLA桌面
蓝白包装物体与CAP起点的配置不同。说明当前全图平均后的余弦距离不足以单独保证
操作状态兼容；不能只取距离分位数就宣称已经校准。缺乏兼容正例，本轮不推荐正式门槛，
`selected_thresholds=null`。保留空间patch信息、分别检查两视角等方向仍需另做验证。

输出（A800代码目录下）：

- `outputs/vision_rms_trial_20261030_v1/`：完整特征、133组距离、网格、36组复核对照图。
- `outputs/vision_rms_visible_reviews_20261030_v1/reviewed_pairs.jsonl`：起点复核与逐条证据。
- `outputs/vision_rms_review_audit_20261030_v1/report.json`：可重复复核审计，无门槛启用。

新增4项针对性CPU测试通过（`test_vision_trial.py`、`test_review_trial.py`）：
有限单位特征、零特征拒绝、窗口去重计数、unknown不等于负例、证据不足不选门槛、
复核身份／重复／跨split拒绝。没有重新运行已通过的整套基线测试。
未启动真实训练、仿真、API调用，未修改原数据或正式训练池。

## 用户接受物体配置差异后的初始门槛选择

用户随后明确允许上述可见物体配置差异，并授权自行确定门槛。已更新
`configs/libero_pi05.json`：coverage为3cm／15°／RMS0.20／图像0.005，
pairing为1cm／5°／RMS0.10／图像0.002，两路图像距离均须通过。
`configs/matching_selection_v1.json`保存选择标准、模型／归一化指纹、实际统计和验证范围。
这些是基于train候选分布选择的初始实验设置，不是已测得准确率最优的校准结果。
此前的36组复核记录保留，其物体配置冲突标签不参与当前门槛筛选或匹配否决。

在原133组几何候选中应用coverage的RMS和图像门槛，保留70组候选关系，涉及27/427
个CAP成功窗口（6.323%）、6/20个原成功片段；train27、validation0、test0。
其中只应用RMS0.20时有84组关系／32窗口，图像门槛进一步保留70组／27窗口。
数量来自已计算候选距离，不代表创建了完整prepared训练池，也不代表匹配准确率。
全部负窗口的严格配对数量尚未统计，pairing值属于更严格的初始设置。

本轮独立报告和候选清单位于 `outputs/matching_threshold_selection_20261030_v1/`。
配置已通过 `validate(matching_ready=True)`；原数据、旧建库和旧试验输出均未改写，未训练。

## 下一位agent的最短复核路径

最新规则优先参见下文“成功片段后缀入池与放宽门槛”；此前逐窗口入池及较窄门槛是历史记录。

1. 读 README、DATA_CONTRACT、AGENTS；确认不是旧SFT脚本。
2. 先阅读上述A800验证报告，不重做已通过的同范围验证；保持原始数据不变。
3. 真实初始化资源未逐条核实，保持按task保守分组；从bank_class派生训练标签，不伪造局部证据。
4. 冻结只输入前视／腕部RGB的编码器；仅train用公开图像和state8校准；CAP成功先匹配VLA成功。
5. 在具备标签和可靠pair后，检查GPU空闲空间，用`--max-updates 1`做真实优化器短更新。
6. 对比短run的恢复、microbatch和CUDA多卡梯度/显存；再由用户启动长训及LIBERO评测。

真实数据导入及GPU0 sampler验收已通过；不得描述为局部负标签已确认、可靠pair已构建，
或真实π0.5训练和LIBERO成功率验收已完成。

## 成功片段后缀入池与放宽门槛（最新）

用户要求：成功代码片段只要有窗口匹配，就从最早匹配窗口起点纳入到该原片段结束；
同时放宽入池，最后明确图像距离0.007。当前preset：

- cap_success_admission=segment_suffix。
- coverage：5cm、30°、RMS0.50、两路图像各≤0.007。
- pairing保持：1cm、5°、RMS0.10、两路图像各≤0.002。

`FrozenMatcher`先完成对冻结VLA正库的所有直接匹配，再按episode＋原segment分组，
从最早匹配点扩展同一成功段的连续后缀。输入顺序不会改变起点；此前前缀、其他程序、
失败代码和未匹配成功段不纳入。后续窗口不作为新的覆盖依据，不会引发CAP链式放行。
coverage明确区分directly_matched和suffix_anchor；保存真实VLA锚点参考与原动作行范围。
prepare／训练正池也采用同一逻辑；新字段参与prepared配置一致性审计，防止沿用旧逐窗资产。
通用Config保持matched_window默认以读取历史资产，当前libero_pi05.json显式选择segment_suffix。

新增可迁移 `success_admission.py` 与 `scripts/run_success_admission.sh`：
原始窗口哈希／特征记录一致性、完整数值候选特征覆盖、后缀动作连续性和原Parquet末尾
均进行检查。缺特征会报错，不会把未编码候选误统计为不匹配。动作payload沿用原窗口。
success-only输出与完整prepared明确区分；未声称已统计全部负配对。

2026-10-30 A800物理GPU0的新screen `pi05_success_suffix_v2`已完成正常退出：
输出 `outputs/success_suffix_run_20261030_v2/`，启动时已保存config快照。

| 统计 | 实际结果 |
| --- | ---: |
| 数值通过的候选关系 | 641 |
| 视觉通过的直接匹配关系 | 501 |
| 自身直接匹配的CAP窗口 | 115 |
| 从匹配点扩展后纳入的CAP窗口 | 182（其中67个无需单独匹配的后续窗口） |
| 纳入后缀的CAP成功原片段 | 14/20 |
| 后缀不重复原动作步数合计 | 1733 |
| VLA正窗口 | 861 |
| 当前正池窗口 | 1043 |

后缀长度42～511动作，14个后缀的末尾均核对到原程序段Parquet末尾。
划分为train182、validation0、test0；没有为凑数跨split匹配。641组候选涉及236个窗口，
编码472张RGB，原生预处理误差0、训练视觉参数0、峰值allocated1.57163GiB，上限8GiB。
原窗口索引SHA未变；未重写原数据、未做仿真／API／训练，也未构建完整负配对prepared。

`configs/matching_selection_v2.json`记录当前选择与实测数量；v1保留为历史设置。
新增7项CPU测试通过：最早匹配起点、未匹配后续动作纳入、前缀／其他程序／失败程序
排除、不链式放行、缺原segment身份不跨窗扩展、动作空洞拒绝、缺特征拒绝、prepared
策略审计、实际Parquet后缀末尾及截断索引拒绝。相关匹配／配置／建库测试亦已通过。
这些是规则与覆盖验证，不能作为匹配准确率或LIBERO恢复成功率。

## 上传前完整CPU回归

2026-10-30，在现有训练仓库的待提交 `pi05_recovery_pair/` 副本中执行完整
`python -m pytest tests`：**68 passed in 20.46s**，包含本机双进程Gloo测试。
40个Python文件通过语法解析，2个Bash脚本通过 `bash -n`；不含GPU／真实训练／仿真／API。
代码和文档收录于训练仓库独立子目录，根目录AGENTS／README提供接手链接；
环境、outputs、数据、模型、特征缓存和私有凭据不纳入提交。

## 负查询仅使用VLA失败片段（最新修正）

用户随后明确要求仅用VLA自主执行失败片段匹配。新增
negative_query_policy=vla_failure_only_v1：matcher只以controller=VLA且原负类标签的
窗口查询正池。CAP失败保留在四类原库／all_windows中，排除在eligible负库、负查询、
training_pool与pair采样外；正参考仍为VLA成功窗口＋已纳入的CAP成功后缀。
成功后缀规则与所有距离门槛均保持原值。

prepared报告及pair记录标注策略；训练前审计拒绝旧的未标注资产和CAP负查询，采样器也
会拒绝CAP失败pair。旧prepared需用当前规则重新prepare，旧原数据不改写。
excluded_negative_windows.jsonl单独记录被排除的CAP失败窗口来源。

2026-10-30 A800仅CPU只读核对原v3窗口：VLA负40窗口／40原片段／40个parent episode，
CAP负11796窗口排除在本轮查询范围。VLA负窗口中39个各含8步有效动作、1个仅含1步；
该1步窗口不满足当前pair至少3步的要求。39只是动作长度满足初始条件的数量，
尚未计算数值／视觉及最终动作mask筛选后的可靠配对数量，不代表39组成功配对。

新增4项负查询策略测试通过；相关匹配／后缀／配置／建库共47项CPU测试通过。
验证了CAP失败完全相同起点也不进入查询、CAP成功仍可作为正参考、原分类与全部来源
保留、prepared及sampler排除CAP失败、CAP失败为唯一负源时FM-only可用，以及旧策略
prepared必须重建。不启用GPU、真实训练、仿真或API。

完整CPU回归为**72 passed in 10.60s**。A800复用已有视觉缓存运行success_admission
完成，输出 `outputs/vla_negative_only_20261030_v1/report.json`：负查询40、排除CAP负11796；
原正池保持861个VLA＋182个CAP窗口（14个成功后缀、1733动作），源索引SHA未变。
这是CPU策略及正池一致性核验，negative_pairing_evaluated=false，未执行完整负配对统计。

## VLA负片段完整匹配统计（2026-10-30）

A800 GPU0实际完成 `python -m recovery_pair.negative_matching`，输出
`outputs/vla_negative_matching_20261030_v1/`；固定当前门槛，未做训练或修改原数据。
源索引SHA256：`bab907ae07282f068a9da56ac550dab2e4dd1427074da1b29be595861fdcf926`。

- 负查询仅VLA失败：40窗口／40原片段／40独立父episode；CAP负11796窗口排除。
- 正池1043窗口：VLA861＋已纳入CAP成功后缀182。
- 严格配对：同公开指令、同split，位置≤1cm、旋转≤5°、state8 RMS≤0.10，
  前视与腕部图像余弦距离各≤0.002；不使用特权信息、任务进度或未来观测。
- 数值通过21个负窗口，30组关系；图像通过仍为21个、30组。
- 动作检查后15窗口／15原片段／15父episode，共21条正参考，全部来自VLA。
  其中train12、validation2、test1；episode各计一次。
- 另6个有状态匹配的负窗口没有可用动作配对。共9组候选关系因
  `insufficient_trusted_overlap`被排除（关系数不等于片段数）：双方有效动作与错误mask
  取交集、移除共享近似动作前缀后，不足3个可比较动作步。
- 为缺少特征的43个候选起点编码86张RGB；冻结原生视觉编码器，无可训练参数；
  原生预处理误差0，CUDA峰值约1.57GiB。
- `report.json`、`negative_window_results.jsonl`、`pairs.jsonl`保存统计与追溯；
  此输出不是完整prepared目录。计数不表示局部负行为已额外验证或任务成功率。

本地完整CPU测试：75 passed in 10.50s，含新增的状态匹配／可训练配对分离、
缺失图像特征拒绝、无数值候选安全返回0测试；未启动真实pair优化器更新。

## 纯正向FM来源差异尾部策略（2026-10-31）

用户确认本轮只用当前已纳入正池进行原生FM训练。新增独立入口
`prepare-positive-fm`、配置`configs/libero_pi05_positive_fm.json`，不修改原匹配库。
纯FM训练必须显式`--mode fm_only`，pair_batch=0、eta=0，负样本与pair均不进入新资产。
不实现此前讨论的单时间点吸引／排斥；旧pair配置保留。

A800实际CPU准备及完整资产审计通过，输出`outputs/positive_fm_source_tail_20261031_v1/`：

- 原正池1043窗口：VLA中途推进528、自主成功333、CAP成功后缀182。
- 舍弃36个不足10真实动作步的VLA中途窗口，保留492个完整窗口。
- 自主成功333和CAP成功182窗口全部保留；共34个成功尾部窗口重复末动作，
  新增185个重复监督动作步。这些重复动作不被表述为真实执行记录。
- 最终1007个正窗口：train VLA646＋CAP182＝828，validation VLA91，test VLA88。
- valid_mask包含成功尾部的重复动作，real_action_mask只标真实记录部分；
  模型25个填充维度始终无效。VLA中途窗口不跨入错误区间、不复制末动作。
- 不增加原始专家40、不重做匹配、不改变原起点采样密度、划分或成功后缀范围。
- 新资产SHA256：`4bd9606ce67f298fccb9987cf4b308f116ccc3ce8fed50bdb78d15b0dcd46cf8`。
- 本次只做真实数据CPU准备与审计；没有真实模型优化器更新、仿真或GPU占用。

本地完整CPU测试80 passed in 10.10s。新增测试覆盖VLA短窗口舍弃、成功末动作复制及其
loss有效性、来源和负类拒绝、尾部损坏审计、纯FM模式约束、一次合成更新及checkpoint恢复。
这不表示真实GPU纯FM更新已验证。默认预算仍为fm_only_updates=100次更新，正式预算可改，
不宣称已实现或执行该数据集的50完整epoch训练。

## 真实GPU0纯FM更新及checkpoint恢复（2026-10-31）

用户授权在GPU0短验证通过后推送。启动前物理GPU0无计算进程，可用约78.7GiB；
PyTorch allocator上限20GiB，入口要求另保留4GiB余量，未kill任何进程。
可复现命令见README的`python -m recovery_pair.fm_validation`；
完成输出`outputs/positive_fm_gpu0_validation_20261031_v2/`，独立screen正常退出。

首次v1在任何优化器更新前发现缓存prefix/suffix的FM与上游原生联合前向不完全一致。
本轮修正真实模型FM路径，直接调用上游`policy.compute_loss`（显式noise/time，
关闭图像随机增强），外部仅按有效mask逐窗口平均；不通过会关闭checkpointing的
`sft_forward`包装器。缓存前缀仍供部署／可微采样，RLinf源码和权重均未修改。
此修正同时适用于旧pair训练的正向FM分支，pair采样定义不变。

- 三类真实训练窗口分别验证：VLA中途10真实步、自主成功8真实＋2重复步、
  CAP成功2真实＋8重复步。原生逐元素loss最大差均为0，梯度有限且非零，冻结参数无梯度。
  旧缓存路径的逐元素loss差分别约0.000323、0.002042、0.091500；不再用其替代原生FM。
- 完整FM64、microbatch1、lr1e-5、clip1、pair0、eta0，执行两次真实优化器更新；
  第一次保存checkpoint后，释放模型，重新加载原始基座，核验初始化哈希，恢复并执行第二次。
- 第1步loss0.11264807，裁剪前梯度范数0.45390469；第2步loss0.08448456、梯度0.62652665。
  固定heldout验证loss依次0.05547954、0.05344647；两步下降不代表已获得任务成功率提升。
- 可训练430098464参数，冻结2923335408参数；两步action expert参数哈希均改变，
  全部冻结参数SHA256始终不变；原始基座文件与归一化资产SHA256也不变。
- 原生／可微sampler前向差0、全部3步梯度检查通过；没有负分支优化。
- 原prepared资产训练后完整审计通过；checkpoint保存和重载恢复通过。
- 峰值allocated15.31037GiB、reserved15.75391GiB，总耗时约211.5秒；任务完成后释放GPU。
- 完整CPU回归81 passed in10.92s，新增测试确保真实FM走原生hook且masked梯度权重正确。

本次仅两次更新和短验证，不是100次正式训练或50epoch实验；未运行LIBERO评测或API。

## 全流程复核与补强（2026-10-31）

本轮检查当前纯FM的准备、来源边界、mask、原生loss、采样、更新、验证、保存和恢复。
修正／补强内容：

- 纯FM的backward=False验证使用no_grad，避免不必要的梯度图；pair验证保留原采样要求。
- 时间采样只设置CPU默认生成器；原torch.manual_seed会额外重置CUDA RNG，而
  fork_rng(devices=[])不会恢复它。保持原Beta分布及同一update的CPU抽样值。
- 纯FM验证覆盖全部heldout正窗口各一次，本批为91窗口，不再只检查随机有放回batch。
- checkpoint新增objective_version=native_fm_masked_window_mean_v1，拒绝把更早的
  缓存FM训练checkpoint静默续到当前目标；另核对sampler位置与update一致。
- 新增可选CPU审计audit --source-norm-stats：逐窗对照原Parquet的执行前state、前视／
  腕部列及row、动作区间与归一化资产；成功重复动作只能起于原段末尾，并核对原payload SHA。

A800重新准备输出`outputs/positive_fm_full_review_prepared_20261031_v1/`完成，1007窗口、
36个中途VLA短尾舍弃、34个成功尾部补齐185步；train828、validation91、test88。
全量对照74个原控制段Parquet及1007个原payload，真实动作最大误差0，执行前状态、
两路图像起点及片段边界均通过。与GPU使用的此前prepared逐条比较，除了输出payload路径，
全部metadata及所有actions/valid_mask/trusted_error_mask/real_action_mask数组完全相同。
证据`outputs/positive_fm_rebuild_equivalence_20261031_v1.json`。

CPU完整回归85 passed in12.07s，覆盖恢复与连续训练精确一致、目标版本拒绝、
完整验证窗口计数、验证不建图、早停、checkpoint保留数量及原数据错位拒绝。

GPU0完整复核输出`outputs/positive_fm_full_review_20261031_v2/`已完成：

- 初次v1在分配模型前因GPU0被其他新进程占用、20GiB上限＋4GiB余量不足而保护退出。
  根据上一轮实测下调上限至17GiB，启动时free21.1796GiB通过检查；未停止其他进程。
- 原始RLinf few-shot基座、FM64/micro1、lr1e-5、clip1、pair0、eta0，共4次真实更新：
  2次组成保存／基座重载／恢复路径，另2次组成从原始基座连续训练的对照路径。
- 两条路径最终全部可训练参数SHA256精确一致，逐步metrics也完全一致；
  全部冻结参数SHA256不变，原始基座和norm资产不变。
- 两步训练loss0.1126480679、0.0844845582；梯度范数0.4539046884、0.6265266538。
  全部91个heldout窗口各检查一次，loss0.0440568280、0.0436320991；
  与旧随机batch验证指标不直接比较，不据此声称LIBERO成功率提升。
- 三类来源的原生compute_loss一致性最大差0，梯度有限且非零；
  原生与可微sampler前向差0、K3全部步骤梯度检查通过。
- 峰值allocated12.51154GiB、reserved12.89258GiB；共享GPU环境下耗时约650秒。
- 训练前后prepared完整审计通过，source alignment全部1007窗口通过。
- 仅4次短验证更新，未启动100更新正式训练、50epoch实验、LIBERO仿真或API调用。


## 完整权重导出／GPU重载精度复核（2026-10-31）

实际执行文档中的`python -m recovery_pair export`，生成6,946,567,178字节的
full_weights.pt。第一次GPU重载发现参数哈希不一致，定位为上游先加载权重、再将主干转BF16、
最后将norm等层转回FP32，丢掉了这些FP32层已训练出的细微更新；并非导出文件缺少参数。

新增checkpoint_io.restore_full_checkpoint_after_dtype_setup，在上游完成dtype布局后，
用原生load_full_weights重新加载完整权重。训练适配与仓库根evaluate.py均采用此处理，
不改RLinf源码或原权重，safetensors few-shot基座加载路径不变。评测身份加入加载策略版本，
旧评测输出需另开目录，不能静默混用。CPU新增回归模拟BF16舍入并验证FP32精确恢复。

GPU0复测输出`outputs/positive_fm_export_review_20261031_v2/`已完成，复用第一次实际export
命令生成的文件，不重复物化；其报告documented_export_cli_passed=false仅表示本次跳过导出，
existing_documented_export_reused=true，不表示导出命令失败。

- 来源base/prepared/mode核对通过；全部可训练与冻结参数哈希均与训练完成时精确一致。
- 原生部署sampler输出[1,10,32]，全部有限；原始基座哈希不变。
- GPU峰值allocated6.63675GiB、reserved6.71484GiB；上限12GiB，额外留4GiB余量。
- 该screen正常完成退出。没有启动LIBERO仿真、付费API或正式训练，没有kill其他进程。
- 完整CPU回归87项通过；另根validate_parallel.py检查2000条评测在1/2/3/16/64/2048
  分片下无重复遗漏，并拒绝不完整／配置不一致结果；此分片检查不使用GPU或真实仿真。
- 已检查Python语法及评测helper导入。验证记录证明当前机器上的短流程，不等于新机器环境
  安装验证、长训稳定性或实际任务成功率保证。


## 全部已准入正窗口用于训练（2026-10-31）

用户要求取消验证／测试划分。新增prepare-positive-fm --all-train，在保持现有已准入池、
动作、mask、窗口起点和成功后缀范围不变的前提下，把生成资产的所有窗口转为train；
source.fm_original_split保留原划分，fm_split_policy=all_train_v1。
不改原库或匹配结果，不因合并划分重新接收CAP片段。
trainer发现没有validation窗口时自动关闭验证与验证早停，仍按显式更新预算结束和保存checkpoint。
完整CPU回归88项通过，新增用例核对全量转入train后所有动作/mask与保留原划分资产一致，
原索引哈希不变，并实际做合成优化器更新检查validation_enabled=false且日志不含validation。
旧GPU报告针对保留划分资产；本次不据此声称新资产已完成真实GPU验证。


## 当前流程入口与A800全train资产（2026-10-31）

新增RUN_POSITIVE_FM.md、scripts/positive_fm.env.example及
scripts/run_positive_fm_pipeline.sh prepare|train|export，明确当前纯FM与历史pair实验区别。
命令串联测试用记录器替代GPU/训练命令，覆盖prepare不启动训练、all-train、视觉单卡、
单卡/多卡启动、恢复/步数限制、导出及GPU数量不一致拒绝；这不是16卡实际训练。
完整CPU回归89项通过，Bash语法及CLI help通过。训练新增可选每rank allocator cap，
要求模型加载前额外4GiB余量；这不限制CUDA上下文/NCCL的额外显存。

A800连接恢复后实际CPU生成outputs/positive_fm_all_train_20261031_v1/并完整审计：
1007个窗口全部train（VLA825、CAP182），validation_enabled=false。
9885真实动作步＋185重复补齐位置＝10070。逐条对照旧资产的动作和全部mask精确一致，
原Parquet state/actions/图像引用对齐通过，最大真实动作误差0。
新资产SHA256为841e044478243ee56facb53222c1ef07c7ea0cced06120a34ff977199df03d68。
没有启动该资产的正式训练或新的GPU短更新；此前GPU0/GPU2结果对应旧保留划分资产。
