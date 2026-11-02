# 方案1恢复成功20条：固定目标反馈补步转换

只读取 `pi05_recovery_raw_scheme1_20261102_v1/recovery_success20_raw` 的固定20条。
成功集的suite配额为 LIBERO-10/Goal/Spatial/Object=3/5/10/2。
不处理失败20、不替换选中轨迹、不使用1200步筛选、不启动训练或模型/API推理。
原始成功但转换失败的也完整保留，标注 `original_success=true`、`converted_success=false`；
不会删掉、改放进既定失败20或自动用别的候选替换。

VLA前缀保留原7维OSC。CAP的每个原始关节动作，用原始状态加官方裁剪后的关节增量
计算固定绝对TCP目标；从转换回放的当前实际位姿计算7维OSC。
平移每轴上限5cm、旋转向量每分量0.5rad。超限就补步，每一步重新读取实际位姿，
继续追同一个目标；最后一个合法动作能完整表达目标时前进，不逐目标等待2mm到位。
每个超限目标最多200控制步；超出预算停止并记为转换失败。

整条从原始初态连续执行，无中间reset、传送或新CAP代码。补步重新记录真实图像和state8。
每步按LIBERO原成功判定检查，首次成功即停止。随后从相同初态仅执行保存的OSC动作，
核对逐步模拟状态误差小于1e-8、前缀一致、成功时刻一致。

`historical_engine/` 是从A800复制的历史控制/记录/回放引擎，控制逻辑未改。
运行器只调用其函数，不调用含旧17条或>1200步筛选逻辑的历史main。
运行器覆盖模块的ROOT/SOURCE/HELPERS路径，文件哈希写入plan.json。
已有同原始动作、同输入文件、同设置且独立回放哈希校验通过的结果直接复用，包括转换失败结果。
旧结果只读，新结果写独立目录。

## A800运行

需已有corrector-eval环境、LIBERO资源、原始数据包，以及原始400条的冻结
`wcm_pi05_monitor/outputs/hybrid_independent_gpu1_20261019_172047/runtime/hybrid` 环境实现。
该runtime及仿真依赖不包含在本目录；这不是独立环境安装包。
启动前检查nvidia-smi。默认物理GPU3只用于EGL渲染，不加载模型；默认2个worker。

```bash
bash run_a800.sh --prepare-only
bash run_a800.sh --episodes 18 --workers 1
# 短样本及独立回放通过后，后台执行全部固定20条；已完成结果校验后复用。
screen -dmS pi05_scheme1_success20 bash -c 'bash run_a800.sh > conversion.log 2>&1'
```

可用 `PI05_ROOT`、`CONVERSION_GPU`、`CONVERSION_PYTHON`、`CONVERSION_RAW`、
`CONVERSION_OUTPUT`、`CONVERSION_WORKERS` 覆盖路径与资源。
输出默认 `$PI05_ROOT/trajectory_conversion_tests/pi05_scheme1_success20_20261102/`：

- `plan.json`：固定20条、输入和代码哈希、可复用结果、明确禁用替换。
- `attempts/episode_*/`：新录制轨迹、视频、7D动作、来源动作和补步映射、复制的CAP程序。
- `converted_episodes/episode_000000`～`episode_000019`：固定20条的独立保存目录，
  成功和失败都保留；复用结果也存到这里。小元数据复制、大录制文件硬链接共享存储，
  不要原地修改大文件；可正常删除旧目录而不影响这里的硬链接。
- `independent_replays/`：独立执行保存动作的逐步核对。
- `results/episode_*.json`：每条最终结果及原版成功程序/转换版首次成功程序对照。
- `status.json`、`summary.json`、`accepted.json`：已完成/成功/失败/技术异常/成功程序不一致统计。

`boundary_audit.converted_program_segments` 按原始代码轮次分段，绝不把相邻代码轮次合并。
`original_success_turn` 与 `converted_first_success_turn` 分开记录；若转换提前在原失败程序成功，
显式标记 `success_program_agrees=false`。成功判定只指转换后的任务结果，
不能据此把整条CAP历史或整段动作都标成优质正监督。

当前输出为可追溯的7D轨迹/图像/state8，`train_ready=false`：尚未做新的局部质量筛选、
分段标签、LeRobot打包或训练库替换。旧引擎report中的 `accepted_for_training` 仅为历史字段，
本次以results/accepted里的转换及独立回放结果为准。
命令支持重复运行完整结果；遇到没有report的中断目录会报技术异常，避免静默覆盖。
