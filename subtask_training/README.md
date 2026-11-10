# 原始22个子任务片段

完整操作入口：[README_SUBTASKS22.md](../README_SUBTASKS22.md)。

只用原来的22段、1311个真实动作，来自12条CAP介入父轨迹；新增46段未纳入。
13个中间片段只保留完整H10窗口，9个最终LIBERO成功片段可按A60重复末动作，共1194窗口。
prompt始终为父轨迹完整instruction。

| 文件 | 用途 |
|---|---|
| fetch.py / build.py | 固定版本原始RGB/state/action物化、按审阅区间切分、逐帧索引 |
| dataset.py | 来源感知的尾部规则；不使用审阅视频作为训练输入 |
| training_data.py | 原A60读取器＋22段；逐字段检查A60配置 |
| plan.py | 真实混合数据、batch及50轮排列的CPU验收 |
| validate.py / check_guards.py | 父行对齐、原始RGB、边界/尾部、多worker、归一化验证 |
| native_probe.py | 现有RLinf环境的CPU原生变换核对，本次暂缓 |
| publish.py | HF上传与全部Git/LFS内容哈希核对 |

requirements.txt仅用于独立CPU数据工具环境。训练复用根目录ENVIRONMENT.md对应环境，
不要用这份清单替换已经验证的训练依赖。数据和权重不包含在代码仓库中。
