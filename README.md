# 基于强化学习的车辆爆胎稳定控制

[中文](README.md) | [English](docs/README.en.md) | [Deutsch](docs/README.de.md) | [한국어](docs/README.ko.md)

本项目使用 CarSim 与 PyTorch 训练连续动作 PPO 策略，在随机车速、路面附着系数、车辆质量、爆胎轮位和爆胎程度下进行爆胎后的轨迹与稳定性控制。当前策略使用特权环境信息训练，并提供基于历史状态序列估计潜变量的适应编码器。

> 本仓库不包含 CarSim、CarSim DLL、`simfile.sim`、本机数据库文件、训练权重或 TensorBoard 日志。运行完整仿真需要有效的 CarSim 安装和本地配置。

## 方法概览

- **策略算法**：PPO + GAE，用于五维连续动作控制。
- **动作空间**：前轮转向角与四轮控制力矩，物理尺度为 `(3, 400, 400, 400, 400)`。
- **有界策略**：高斯潜在动作经过 `tanh` 和物理尺度映射；保存 pre-tanh 动作用于一致的 PPO log-prob 计算。
- **探索尺度**：每个动作维度使用全局可学习的 `log_std`，并限制在设定范围内。
- **特权训练**：爆胎轮位/程度、附着系数和质量经过 privileged encoder 得到潜变量。
- **历史适应**：一维 CNN 从 25 步状态历史预测 privileged latent，支持不直接访问环境参数的闭环控制。
- **课程学习**：随 episode 增加车速、低附着与爆胎程度的难度范围。

## 项目结构

```text
tire_burst/
├── generated_tire_tables/     # 25/50/75/100% 爆胎 Fx/Fy/Mz 表及清单
├── scripts/
│   ├── main.py                # PPO 训练、续训、测试和数据采集入口
│   ├── train_history_encoder.py
│   └── compare_oracle_history.py
├── src/
│   ├── agents/                # PPO、特权编码器和历史编码器
│   ├── envs/                  # CarSim 包装器与爆胎环境
│   ├── models/                # 状态、reward、动作尺度和车辆参数
│   ├── mpc/                   # LPV-MPC 参考实现
│   └── utils/                 # VS Solver、轮胎表生成/替换工具
├── tests/                     # 不依赖完整 CarSim 运行的单元测试
├── docs/                      # 英文、德文、韩文 README
├── TRAINING_COMMANDS.md       # PowerShell 命令手册
└── README.md
```

本地运行会另外产生 `checkpoints/`、`datasets/`、`runs/` 和 `simfile.sim`；这些均已由 `.gitignore` 排除。

## 环境要求

- Windows
- Python 3.10 或更高版本
- CarSim 2023（当前代码按 CarSim 2023.2 DLL/数据库接口开发）
- PyTorch、NumPy、TensorBoard
- CasADi（仅 LPV-MPC 相关模块需要）
- `pefile`、`tomlkit`

示例安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch tensorboard casadi pefile tomlkit pytest
```

本项目暂未提供锁定版本的依赖文件。正式复现实验时应记录 Python、PyTorch、CUDA、CarSim 与依赖版本。

## CarSim 本地配置

1. 将有效的 `simfile.sim` 放到项目根目录。
2. 根据本机数据库修改以下文件中的 CarSim 绝对路径：
   - `src/envs/tire_burst_env.py`
   - `src/utils/generate_burst_force_tables.py`
   - `src/utils/get_dll_functions.py`
3. 确认 `.sim` 导入/导出通道与代码一致：6 个 import 通道，并至少提供 8 个 export 变量。
4. 训练会把抽取到的爆胎程度表写入配置的 CarSim 轮胎数据集。使用前请备份原数据库。

仓库中的绝对路径是开发机示例，不保证在其他机器上可用。`generated_tire_tables/manifest.json` 中的路径用于记录数据来源，不是跨机器配置文件。

## 快速开始

在项目根目录执行测试：

```powershell
python -m pytest -q
```

从头训练 PPO：

```powershell
python -m scripts.main --mode train --max_episodes 4000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
```

从 checkpoint 续训：

```powershell
python -m scripts.main --mode resume --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
```

Oracle 确定性测试：

```powershell
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_squashed.pth"
```

更多训练、历史数据采集、编码器训练、配对评估和 TensorBoard 命令见 [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md)。

## 历史适应流程

采集 Oracle latent 数据：

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_squashed.pth" --adaptation_episodes 500 --dataset "datasets\adaptation.pt"
```

训练历史编码器：

```powershell
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history_encoder.pth" --epochs 30 --batch_size 64 --learning_rate 0.0005
```

比较 Oracle 与历史策略：

```powershell
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_squashed.pth" --history_checkpoint "checkpoints\history_encoder.pth" --episodes 20
```

## 爆胎表工具

重新生成爆胎程度表：

```powershell
python -m src.utils.generate_burst_force_tables --levels 0.25 0.5 0.75 1.0 --output-dir "generated_tire_tables" --reference-dir "generated_tire_tables\burst_075" --reference-severity 0.75 --mz-full-burst-scale 0.25
```

将 100% 爆胎表写回配置的 CarSim 数据集：

```powershell
python -m src.utils.restore_full_burst_tables --table-dir "generated_tire_tables\burst_100"
```

第二个命令会修改本机 CarSim 数据库，请确认目标路径并提前备份。

## Reward 与实验解释

reward 定义位于 `src/models/reward_model.py`，包含速度、轨迹、稳定性、控制幅值、动作平滑性和异常终止分量。更改 reward 后继续训练等价于从旧策略初始化一个新优化目标；新旧 reward 下的 episode return 不应直接比较。

推荐在固定、独立、配对的工况集合上比较原始物理指标，包括轨迹误差、横摆响应、侧向速度、动作 RMS/饱和率、失败率和最差分位数。当前入口尚未提供完整的命令行随机种子控制，正式统计实验应补充统一 seed 管理。

## 不提交到 GitHub 的内容

`.gitignore` 排除了：

- `runs/` TensorBoard 日志；
- `checkpoints/`、`datasets/` 和模型/数据文件；
- `.sim`、`.par`、`.dll` 等本地或专有 CarSim 文件；
- Python 缓存、测试缓存、虚拟环境、日志和编辑器配置。

若需要发布模型，请使用 GitHub Release、外部模型仓库或 Git LFS，并同时提供对应代码提交、reward 版本、训练命令和评估结果。

## 当前状态

这是研究原型，不是量产车辆控制软件。策略仅应在受控仿真环境中使用；任何实车实验都需要独立的安全约束、故障保护、形式化验证和硬件在环测试。

## 许可证

仓库当前尚未包含许可证。公开发布前请添加适合你的 `LICENSE`；在此之前，默认版权仍由作者保留，第三方没有自动获得复制、修改或分发许可。
