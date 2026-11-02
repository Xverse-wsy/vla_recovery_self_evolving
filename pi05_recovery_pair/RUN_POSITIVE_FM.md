# 正向FM操作入口

当前方案1只使用 **[README_STAGE1.md](README_STAGE1.md)**：包含阶段目标、实际数据量、FM公式、安装、严格建库匹配、GPU预算、启动、恢复、导出和验收。

当前是13,407窗口、全局batch128、完整50epoch；VLM不冻结，原生预处理和H10×32 FM。严格匹配当前准入CAP为0，不能使用历史14段CAP/16,324窗口的数字启动本轮。

- 新机严格建库：`recovery_conversion/build_scheme1.py --stage banks|match|prepare|align-training`，每阶段单独执行。
- 已有本机资产：原位复用prepared，在独立run中调用 `scripts/run_positive_fm_pipeline.sh train|export`。
- `scripts/positive_fm.env.example` 提供机器参数模板；本轮的PAIR_RUN/数据路径按当前主文档设置。
- 通用脚本的 `prepare` 是历史宽松匹配入口，不用于本次严格方案1。
- 原始转换/切分来源与工具边界见 [README_SCHEME1_STAGE1.md](../recovery_conversion/README_SCHEME1_STAGE1.md)。历史实测见 [VALIDATION.md](VALIDATION.md)。

Stage2与A60/B80总实验使用独立入口，不在此流程中自动运行。
