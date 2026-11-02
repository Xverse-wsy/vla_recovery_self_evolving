# Stage1/Stage2接手入口

当前操作文档是 [README_STAGE1.md](README_STAGE1.md)；来源与重建工具见 [README_SCHEME1_STAGE1.md](../recovery_conversion/README_SCHEME1_STAGE1.md)，验证见 [VALIDATION.md](VALIDATION.md)。先按当前文档操作，不从README后半的历史pair实验启动本轮。

## 最新授权

2026-11-03：对齐Stage1到A60、检查代码/文档、push GitHub；A800只做最多2次更新的GPU测试，不正式长训。测试保持global128，用micro1和GPU0 allocator40GiB，启动前确认额外4GiB余量。绝不kill他人进程。后续正式50epoch需明确授权。

## 当前Stage1规则

- `stage1_a60_sft_v1`，VLM和其余参数全部训练；原生compute_loss(train=True)、H10×32均值，包含25个模型padding维度。真实动作7维。FM只随机采一个flow time，不反传采样链。
- 原始few-shot权重、norm/tokenizer初始化。config/base.yaml共享A60优化器/调度；lr5e-6、β(.9,.95)、eps1e-8、wd1e-10、clip1、warmup100、cosine到5e-7。推理K5 Euler。
- 默认global128/micro2；显存有限可micro1。每轮全正池打乱无放回，完整50epoch，无来源配额、pair、验证/测试或早停。尾batch按实际窗口数归一化，不丢/复制/跨轮补批。
- 当前方案1窗口13407/去重动作13767；专家6632、自主3232、VLA推进3543窗口/3903帧。CAP严格准入0。旧16324/14段CAP/冻结配方属历史数据，不混用。
- VLA推进的segmented记录已物理去掉最后10动作和起点行，不重复裁剪；训练仍要求完整真实H10，不补齐。专家/完整自主成功/准入CAP尾部可重复末动作并参与FM。
- 先由原selection/provenance建四类库，不重新发明局部标签。仅原始/转换均成功且成功代码轮次对应的CAP片段可申请准入；原分类和失败样本完整保存。
- CAP查询冻结的VLA正库。严格1cm/5°/state RMS.10/每指5mm/两路图像各.002，最多50数值候选。只使用同公开指令、动作前state8及前视/腕视RGB；不使用进度、私有task/object ID、对象位姿或持物/接触真值。专家只在匹配后加入FM。
- 成功代码段最早匹配起点放行同段到成功的后缀，不跨程序/交接/标签边界；不能声称后续帧独立匹配。code不能链式为其他code放行。
- 匹配stride10；准入后训练stride1。新机从完整HF数据目录重建本机绝对路径索引，不直接训练A800索引。使用完整repo，独立子目录缺config/base.yaml会拒绝执行。
- 库与历史资产不可变；训练写独立run，可以原位只读复用prepared。换数据/global batch/目标/预算用新run，不resume旧冻结/稀疏/batch2验证checkpoint。microbatch可调但不承诺不同硬件逐位复现。

## 代码与检查

新机严格流程用../recovery_conversion/build_scheme1.py的banks→match→prepare→align-training。仅从HF迁移不运行原始export/转换工具。run_positive_fm_pipeline.sh的通用prepare是旧宽松配方；当前只用它的train/export入口。

`openpi_backend.py` 必须保留原生joint prefix/suffix FM。缓存前缀只用于旧采样路径，不替代FM。完整权重加载保留checkpoint_io的dtype重载修复。手动SUM归约已经按global batch缩放，不再叠加DDP。

测试：python -m pytest（本机Gloo需要IPC/回环权限）。本机缺少RLinf/OmegaConf的检查需A800真实环境完成。GPU检查核验原生FM、视觉梯度、checkpoint/连续训练一致性。不能以低batch短测代替生产batch检查或完整50轮。

## 旧Stage2保护

Stage2模块/配置/旧资产与legacy配置身份不改。旧Stage2仍用VLA失败查询、numeric20门槛2cm/10°/RMS.20、图像.002、至少3有效步；历史144负窗口/26父轨迹/205去重动作只适用于旧池。CAP失败保留分类但不进入负查询。

其完整可微Euler、同mask吸引/持续排斥loss、0<lambda_push<1、无hinge/截止门槛、从纯噪声启动及不detach采样链等语义保留。当前新Stage1 checkpoint不能直接接旧Stage2，需要另行适配；不得绕过objective_version或初始化/50epoch校验。操作见README_STAGE2.md和README_STAGE2_MATCHING.md，本轮不启动它们。

密钥/数据/权重/缓存/输出不入Git；不启动付费API或长评测，不改原始数据，不终止他人进程。任务成功只认LIBERO，代码无报错不等于任务成功。
