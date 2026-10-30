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
