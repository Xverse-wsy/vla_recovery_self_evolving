# Agent接手入口

## 新增实验：A60＋原始22个子片段（2026-11-10）

先读 [README_SUBTASKS22.md](README_SUBTASKS22.md)。用户改回原22段，新增46段不纳入。
实验a60_subtasks22：专家40＋原自主20＋22段，11058窗口，完整50epoch。
新YAML直接继承A60，模型/优化器/调度/运行参数启动时逐字段核对。
13个中间片段仅起点0..n-10，不重复末动作；9个最终LIBERO成功段可按A60补齐。
prompt必须用父轨迹完整instruction；不再额外物理删除段末10步。
保留A60/B80及原Stage1行为；direct_data.build_raw只为新kind选择新读取器。
最新用户已授权CPU测试、HF上传、GitHub推送和A800有界GPU短测试；
先查当前空闲显存，不影响其他进程，不启动正式50轮。2026-11-10连接恢复后，复用A800
12条哈希一致的转换父轨迹恢复原22段；全部训练列与HF版一致。HF HTTPS仍有TLS错误。
本机/A800各32项CPU测试、完整数据验收和原生5样本变换通过；GPU4的10次更新及第2步
恢复重放8次通过，最终权重/优化器/输入计数一致。峰值allocated32.363/reserved33.973GiB，
allocator34GiB/headroom8GiB；测试已退出释放GPU，原有进程均仍在。完整50轮未启动。
短测保留全部早期checkpoint供第2步恢复，正式最近3份策略和A60配方保持原行为。
数据/权重不入Git。HF新增目录pi05_subtasks22_20261110。
CPU入口：python -m subtask_training.plan --output /new/report；测试：python -m pytest tests -q。

## 现有方案1 Stage1（2026-11-03，独立实验）

先读 [pi05_recovery_pair/README_STAGE1.md](pi05_recovery_pair/README_STAGE1.md)。这是当前方案1从部署、建库到训练/恢复/导出的唯一操作说明。根目录A60/B80和旧Stage2是独立实验，不能混用其数据量或启动命令。

主说明已包含HF实际目录映射、固定数据revision下载、本机重建、建库计数验收、2步短测试及CPU checkpoint重载检查。部署/核查任务按短测试入口执行，到验收停止；正式50epoch命令单独列出，不因按顺序读文档而自动执行。

用户补充：目标机已经完成A60、B80和旧Stage1。优先复用现有环境和原始资源，先按主说明末尾“已有训练机的增量迁移与验收”检查；不盲目重装共享环境、不覆盖旧数据/库/checkpoint。新版四个方案1数据目录不能用旧B80 recovery20代替，旧Stage1权重不能作为本轮初始化或续训。

当前用户授权：复核代码/文档、push GitHub、在A800做有界GPU测试。**不启动正式50轮训练**。最近测试使用GPU0、allocator40GiB、global batch128/microbatch1，最多2次更新。先查nvidia-smi，至少留额外4GiB，不kill他人进程。正式训练命令仅供用户后续显式启动。

- 配方 `stage1_a60_sft_v1`：全参数训练、VLM不冻结、原生训练预处理及完整H10×32 FM均值；真实环境动作7维。A60共用config/base.yaml与原生AdamW/调度：lr5e-6、β=(.9,.95)、eps1e-8、weight_decay1e-10、clip1、warmup100、cosine到5e-7。
- 全局batch128、默认microbatch2，可按共享卡显存改micro1；全部正窗口每轮无放回遍历，完整50epoch。无pair、无负训练、无验证/测试或早停。A60 YAML batch回退值100，16卡示例128；两个入口均可配置，不能误称默认值完全相同。
- 方案1当前13,407窗口、13,767去重真实动作；专家6632、自主成功3232、VLA推进3543窗口/3903帧。严格CAP准入为0，不伪造恢复监督。B128对应105更新/轮、尾批95、50轮5250更新。
- 当前资产 `dataset_subsets/pi05_recovery_scheme1_stage1_20261102/stage1_a60_sft_v1/`；旧父目录algorithm/prepared及旧16,324窗口资产保留溯源，不作为当前启动依据。索引含本机绝对路径，新机器必须重建。
- 原始成功20转换后12成功/8失败全部保留；原始失败20的转换标签另存；两条只有CAP启动失败的VLA前缀，不能伪造CAP动作或完整回放。
- 已切分的VLA推进段已实际删除末10个动作及起点行，读取时不重复删除；窗口还需完整真实H10、不补齐。专家/完整自主成功/准入CAP成功尾部允许重复末动作，重复参与FM。
- CAP只匹配原始/转换均成功且成功代码轮次对应的片段。门槛1cm/5°/state RMS.10/每指5mm/前视与腕视各.002，最多50候选。仅模型可见指令、执行前state8和RGB；不用进度、对象位姿、持物/接触真值或未来观测。
- 最早匹配点可纳入同一实际成功代码段到成功的完整后缀，不跨程序。专家只在匹配后进入FM，不进入匹配库。

## 入口和修改位置

- 新机从HF四个完整/segmented数据目录建库：`recovery_conversion/build_scheme1.py --stage banks|match|prepare|align-training`，按主说明分别运行。
- `run_scheme1_build.sh` 含原始导出/等待转换；仅迁移现有HF数据时不要运行它。转换/切分的历史依赖边界见 [README_SCHEME1_STAGE1.md](recovery_conversion/README_SCHEME1_STAGE1.md)。
- `pi05_recovery_pair/scripts/run_positive_fm_pipeline.sh train|export` 复用当前prepared；通用prepare使用历史宽松门槛，不用于本次严格方案1。
- 路径/GPU/输出：shell环境；batch/microbatch：新run的algorithm.json；优化器和调度：config/base.yaml。默认模型/epoch/lr/clip/seed与JSON不一致会拒绝启动。
- 环境：Linux/Python3.12，完整训练依赖见ENVIRONMENT.md、install_environment.sh train。子目录pyproject不是完整训练环境。固定RLinf c70606f08cdca259b8dec03d4430926b5b8fac9d，需要可追溯Git源码。
- GPU报告与历史验证：pi05_recovery_pair/VALIDATION.md。短验证不等于正式50轮/H10016卡/任务成功率通过。
- HF当前配方已发布并核验，提交aced94d551789d25a7a66af329f9bdcf1b232c2c。HF代码补丁供旧clone使用；当前GitHub主线不再覆盖旧补丁。旧stage1_libraries.tar.gz为冻结配方。

## Stage2兼容边界

Stage2代码、配置与既有数据保持原有行为，旧入口不能直接接受此版全参数Stage1 checkpoint。未来接入需单独适配和授权；不能为接收新权重而绕过目标版本校验。
保留完整权重加载的dtype重载修复，防止FP32层中间BF16舍入。改变共享执行代码须保留legacy配置身份与数值行为。手动梯度SUM不能再叠加DDP平均。

## 根目录A60/B80（独立历史对照）

A60=专家40+自主20，B80=同60+旧恢复20；各从原始few-shot初始化、完整50轮。原位读取Parquet，不合并数据、不改原norm。原三模型完整评测为Spatial/Object/Goal/LIBERO-10各10任务×50初态，每模型2000条。训练/评测总入口见RUNNING.md；环境见ENVIRONMENT.md；根目录run_experiment.sh不用于当前片段Stage1。
相关代码：config/base.yaml、config/a60.yaml、config/b80.yaml、train.py、direct_data.py、training_runtime.py、evaluate.py/evaluate_parallel.py。配置和launcher batch需查看实际生效值。

## 通用约束

密钥从私有环境/缓存读取，不写跟踪文件；数据、权重、环境和大输出不入Git。只检查/部署/上传时不隐式启动长训、长评测或付费API。输出只写新目录，不覆盖原数据或历史结果。GPU空闲不等于资源预留，检查后再分配，不停止他人进程。
修改后做Python/Bash语法、git diff --check及相关测试。报告真实覆盖范围；不把模拟CPU检查写成真实GPU训练，不把采样50轮写成已训练50轮。单机torchrun standalone支持单卡/多卡，未验证H10016卡或跨节点。
