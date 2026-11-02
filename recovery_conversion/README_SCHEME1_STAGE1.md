# 方案1恢复40条：来源、转换与严格建库

当前训练的唯一操作说明是 [README_STAGE1.md](../pi05_recovery_pair/README_STAGE1.md)，其中列出安装、数据目录、迁移重建、训练命令、GPU限制、FM公式、完整50轮预算、恢复和导出。

## 本轮来源与保留规则

固定原始成功20条、原始失败20条，来自已确认的400条评测。成功来源20转换后12成功/8失败，全部保留，不替换。失败来源20按原始执行结果分组，转换后成败另记；其中两条只有CAP启动失败的VLA前缀，没有CAP动作，不伪造完整回放结果。

完整父数据与切分结果分别保存为 `recovery_success20`、`recovery_failure20`、`recovery_success20_segmented`、`recovery_failure20_segmented`。这四个HF目录均已上传。分类使用既有WCM边界和每轮完整代码；任务完成只认LIBERO transition。VLA推进片段在切分导出时实际删除末10个动作及起点行，删除范围留审计；完整父轨迹仍保留原记录。

CAP正准入要求原始/转换均成功且成功代码轮次对应；同一程序最早匹配点至实际成功的后缀才可入池。严格门槛为位置1cm/旋转5°/state RMS0.10/每指5mm/两路图像各0.002，只用模型可见观测。当前没有CAP后缀通过；训练池共13,407窗口（专家6632、自主3232、VLA推进3543），13,767去重真实帧。全局128时每轮105更新、尾批95，完整50轮5,250更新。

## 两类工具的边界

`convert_selected_success.py`、`convert_fixed_failures.py` 和 `build_scheme1.py --stage export` 用于从原始混合记录转换、回放/导出，依赖A800历史runtime、环境和原始sim资源；它们不是跨机器一键安装包。`run_scheme1_build.sh` 包含导出及等待转换逻辑，不能拿来训练或在仅迁移已导出数据时重复运行。

新机已有HF四个数据目录时，只运行 `banks → match → prepare → align-training`，按主说明重建本机索引，不重做转换/回放。`align-training` 生成独立 `stage1_a60_sft_v1/`，逐窗确认起点、动作、mask和准入范围不变；拒绝覆盖已完成输出。

A800根目录：`/home/wenchaoxu/wsy/vla-recovery-posttrain-loop`。
适配代码：`training_configs/pi05_ablation_direct_20261029/`。
当前资产：`dataset_subsets/pi05_recovery_scheme1_stage1_20261102/stage1_a60_sft_v1/`。
必须使用完整适配repo，旧独立 `training_configs/pi05_recovery_pair/` 缺共享config不能启动本配方。

## HF和迁移

[HF目录](https://huggingface.co/datasets/wsy-w1k/rollout/tree/main/pi05_recovery_scheme1_20261102) 已发布6个新版文件，提交 `aced94d551789d25a7a66af329f9bdcf1b232c2c`，远端size/SHA256核验通过。

- `stage1_a60_sft_v1.tar.gz`：A60配方训练资产和验证报告，索引含A800路径，不能直接在新机训练。
- `stage1_training_patch_a60_sft_v1.tar.gz`：供此前旧clone使用的代码补丁，解压到完整适配repo根目录；当前GitHub主线无需再应用旧补丁覆盖。
- `rebuild_scripts_a60_sft_v1.tar.gz`：该发布快照的建库工具。当前GitHub的同名目录与文档为后续修订入口。
- `active_training.json`、`stage1_a60_report.json`、`README.md`：资产入口和审计。

旧 `stage1_libraries.tar.gz` 含历史冻结配方，保留溯源，不用于当前训练。下载原始/segmented目录与训练权重，克隆完整当前代码，按主文档在新工作根目录重建。不要覆盖旧资产，也不要从旧冻结/验证batch2的checkpoint续正式batch128。

## 验收与正式训练

GPU0有界验证4次更新通过，原生FM及视觉梯度正常，连续/断点恢复权重与日志精确一致；峰值32.42GiB。验证batch2/micro1，正式全局batch128。完整50轮与任务成功率不得用短验证代替。

本轮只授权A800有界GPU测试和代码推送，不启动正式50轮。具体GPU、显存快照和参数记录在独立training_runs目录；建库资产的 `training_started=false` 仅表示建库过程未训练，不代表后续测试run状态。
