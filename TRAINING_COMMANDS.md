# 训练与评估命令手册（PowerShell）

本文档中的命令均按当前项目代码编写，可在 PowerShell 中直接复制执行。

## 0. 进入项目目录

每次打开新的 PowerShell 后先执行：

```powershell
Set-Location "C:\path\to\tire_burst"
```

创建文档命令使用的输出目录：

```powershell
New-Item -ItemType Directory -Force -Path "checkpoints","datasets" | Out-Null
```

确认 Python 和 CUDA 是否可用：

```powershell
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

运行项目单元测试：

```powershell
python -m pytest -q
```

> `scripts.main` 会自动选择 CUDA；没有可用 CUDA 时使用 CPU。运行训练前需要保证 `simfile.sim`、CarSim DLL，以及代码中使用的 CarSim 数据库路径均可访问。

## 1. 推荐的完整训练流程

### 1.1 使用新 reward 从头训练 PPO

下面的命令训练 8000 个 episode，每 100 个 episode 保存一次，并在结束时保存最终模型：

```powershell
python -m scripts.main --mode train --max_episodes 8000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --actor_learning_rate 0.0005 --critic_learning_rate 0.001 --gamma 0.995 --lmbda 0.95 --eps_clip 0.2 --K_epochs 3 --entropy_coef 0.001
```

快速冒烟训练（只用于检查流程，不用于评价策略）：

```powershell
python -m scripts.main --mode train --max_episodes 8 --checkpoint_interval 4 --checkpoint "checkpoints\ppo_smoke_test.pth"
```

注意：

- `train` 从随机初始化开始。
- 如果指定的 checkpoint 已存在，训练过程中及训练结束时会覆盖该文件。
- `checkpoint_interval` 只是定期刷新同一个文件，不会自动生成并保留每个历史 checkpoint。需要历史快照时应在训练期间另行复制并改名。
- 当前 PPO 每收集 4 个 episode 更新一次，因此 `max_episodes` 最好是 4 的倍数。
- 训练工况使用 curriculum；episode 编号会影响课程难度。

### 1.2 从旧模型切换到新 reward 后继续训练

先复制旧 checkpoint，避免覆盖旧模型：

```powershell
Copy-Item "ppo_checkpoint_latent_8000.pth" "checkpoints\ppo_reward_v2_finetune.pth"
```

在复制后的 checkpoint 上继续训练 2000 个 episode：

```powershell
python -m scripts.main --mode resume --max_episodes 2000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_reward_v2_finetune.pth" --actor_learning_rate 0.005 --critic_learning_rate 0.0001 --gamma 0.995 --lmbda 0.95 --eps_clip 0.2 --K_epochs 2 --entropy_coef 0.001
```

`--max_episodes 2000` 表示本次额外训练 2000 个 episode，而不是训练到累计第 2000 个 episode。`resume` 会恢复 actor、critic、privileged encoder 及两个优化器的状态，并从 checkpoint 记录的 episode 继续 curriculum。

重要：PyTorch 在恢复优化器状态时会同时恢复 checkpoint 中的学习率，因此 `resume` 命令里的 `--actor_learning_rate` 和 `--critic_learning_rate` 会先用于构造优化器，随后通常被 checkpoint 内保存的旧学习率覆盖。当前脚本没有“恢复后重设学习率”的参数；若必须更改续训学习率，需要修改加载逻辑或加载后手动更新 optimizer 的 `param_groups`。

> 统计对照建议同时保留“新 reward 从头训练”和“旧 checkpoint 上微调”两条实验线，不能只比较两套 reward 下的 return 数值。

### 1.3 测试 Oracle 策略

Oracle 策略可以直接访问爆胎轮位、爆胎严重度、路面附着系数和车辆质量等特权信息。

测试从头训练模型：

```powershell
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_reward_v2_fresh.pth"
```

测试微调模型：

```powershell
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_reward_v2_finetune.pth"
```

当前 `test` 模式只运行 1 个随机 episode，适合功能检查，不足以支持统计结论。多工况配对比较使用第 5 节的 `compare_oracle_history.py`。

### 1.4 采集历史编码器训练数据

使用已经训练好的 PPO Oracle 策略采集 500 个 episode：

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --adaptation_episodes 500 --dataset "datasets\adaptation_reward_v2.pt" --history_checkpoint "checkpoints\history_reward_v2.pth"
```

数据内容为 25 步状态历史及对应的 Oracle latent `z`。采集使用完整随机工况分布，不使用 curriculum。

`collect_history_data` 实际使用 PPO 的 privileged encoder 生成标签；历史编码器 checkpoint 不参与标签生成。因此历史编码器文件尚不存在时出现“random CNN”提示不会改变采集标签。

### 1.5 从头训练历史编码器

```powershell
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation_reward_v2.pt" --checkpoint "checkpoints\history_reward_v2.pth" --epochs 30 --batch_size 64 --learning_rate 0.0005
```

### 1.6 续训历史编码器

下面的命令在已有权重上额外训练 15 个 epoch：

```powershell
python -m scripts.train_history_encoder --resume --dataset "datasets\adaptation_reward_v2.pt" --checkpoint "checkpoints\history_reward_v2.pth" --epochs 15 --batch_size 64 --learning_rate 0.0002
```

注意：该脚本只恢复网络权重，不恢复 Adam 优化器状态和历史 epoch 编号，属于重新创建优化器后的微调。

### 1.7 测试历史编码器的 latent 拟合误差

```powershell
python -m scripts.train_history_encoder --test --dataset "datasets\adaptation_reward_v2.pt" --checkpoint "checkpoints\history_reward_v2.pth"
```

当前命令在传入的整个 dataset 上计算 MSE。如果该 dataset 同时用于训练，这只是训练集拟合误差，不能作为泛化误差。严谨实验应另建独立验证/测试 dataset。

### 1.8 使用历史编码器闭环测试

运行 20 个随机 episode：

```powershell
python -m scripts.main --mode test_history_encoder --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --history_checkpoint "checkpoints\history_reward_v2.pth" --adaptation_episodes 20
```

这里 `--adaptation_episodes` 在该模式下表示测试 episode 数，而不是训练轮数。

## 2. PPO 各模式命令模板

### train：随机初始化训练

```powershell
python -m scripts.main --mode train --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_new.pth"
```

### resume：从完整 PPO checkpoint 续训

```powershell
python -m scripts.main --mode resume --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_existing.pth"
```

### test：Oracle 确定性策略测试

```powershell
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_existing.pth"
```

### collect_history_data：采集适应网络数据

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_existing.pth" --adaptation_episodes 500 --dataset "datasets\adaptation.pt"
```

### test_history_encoder：历史估计 latent 的闭环测试

```powershell
python -m scripts.main --mode test_history_encoder --checkpoint "checkpoints\ppo_existing.pth" --history_checkpoint "checkpoints\history_existing.pth" --adaptation_episodes 20
```

## 3. PPO 超参数模板

当前代码的默认值为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--actor_learning_rate` | `0.005` | actor 和 privileged encoder 学习率 |
| `--critic_learning_rate` | `0.0001` | critic 学习率 |
| `--gamma` | `0.995` | 折扣因子 |
| `--lmbda` | `0.95` | GAE lambda |
| `--eps_clip` | `0.2` | PPO clipping 范围 |
| `--K_epochs` | `2` | 每次 PPO 更新的训练轮数 |
| `--entropy_coef` | `0.001` | 熵奖励系数 |
| `--max_episodes` | `1000` | 本次运行的 episode 数 |
| `--checkpoint_interval` | `100` | checkpoint 保存间隔；`0` 表示只在最终保存 |
| `--adaptation_episodes` | `100` | 数据采集或历史策略测试的 episode 数 |

完整自定义模板：

```powershell
python -m scripts.main --mode train --actor_learning_rate 0.005 --critic_learning_rate 0.0001 --gamma 0.995 --lmbda 0.95 --eps_clip 0.2 --K_epochs 2 --entropy_coef 0.001 --max_episodes 8000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_custom.pth"
```

## 4. 历史编码器脚本命令模板

从头训练：

```powershell
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history.pth" --epochs 15 --batch_size 32 --learning_rate 0.0005
```

续训：

```powershell
python -m scripts.train_history_encoder --resume --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history.pth" --epochs 15 --batch_size 32 --learning_rate 0.0005
```

只测试：

```powershell
python -m scripts.train_history_encoder --test --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history.pth"
```

不要在同一次调用中同时传入 `--test` 和 `--train`，否则脚本会先测试再继续训练。

## 5. Oracle 与历史策略配对比较

该脚本会先在一个工况下运行 Oracle，然后在完全相同的工况下运行 history 策略，同时报告 latent RMSE 和两者 return。

### 5.1 随机工况比较

```powershell
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --history_checkpoint "checkpoints\history_reward_v2.pth" --episodes 20
```

### 5.2 固定困难工况比较

下面示例表示初速 140 km/h、附着系数 0.4、质量 1500 kg、0.75 爆胎严重度、0 号轮爆胎：

```powershell
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --history_checkpoint "checkpoints\history_reward_v2.pth" --episodes 10 --fixed --v 140 --mu 0.4 --m 1500 --burst_severity 0.75 --burst_tire 0
```

测试完全爆胎工况：

```powershell
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_reward_v2_fresh.pth" --history_checkpoint "checkpoints\history_reward_v2.pth" --episodes 10 --fixed --v 150 --mu 0.3 --m 2000 --burst_severity 1.0 --burst_tire 0
```

轮位编码为 `0、1、2、3`。固定工况下重复多个 episode 的动力学和确定性动作相同，因此它们不构成独立统计样本；统计比较应改变速度、附着系数、质量、严重度和轮位，而不是只重复同一点。

## 6. TensorBoard

训练日志默认写入项目根目录下的 `runs`。启动 TensorBoard：

```powershell
python -m tensorboard.main --logdir "runs" --port 6006
```

然后在浏览器打开：

```text
http://localhost:6006
```

主要指标：

- `Reward/Episode_Return`
- `RewardComponent/speed`
- `RewardComponent/track`
- `RewardComponent/stability`
- `RewardComponent/control`
- `RewardComponent/smoothness`
- `RewardComponent/terminal`
- `Loss/Actor`
- `Loss/Critic`
- `Policy/Entropy`
- `Policy/ClipFrac`
- `Policy/NormalizedStd`
- `Action/steer`、`Action/L1`、`Action/R1`、`Action/L2`、`Action/R2`

不同 reward 的 return 不能直接横向比较。对新旧 reward 应比较相同评估工况下的原始物理指标及各 reward 分量。

## 7. 轮胎爆胎表工具

项目已经包含 `generated_tire_tables`，通常不需要在每次训练前重新生成。

重新生成 25%、50%、75%、100% 爆胎表：

```powershell
python -m src.utils.generate_burst_force_tables --levels 0.25 0.5 0.75 1.0 --output-dir "generated_tire_tables" --reference-dir "generated_tire_tables\burst_075" --reference-severity 0.75 --mz-full-burst-scale 0.25
```

将修正后的 100% 爆胎表恢复到代码配置的 CarSim 目标数据集：

```powershell
python -m src.utils.restore_full_burst_tables --table-dir "generated_tire_tables\burst_100"
```

第二条命令会写入 CarSim 数据库中的目标轮胎文件，只应在确认目标路径正确且确实需要恢复时运行。训练环境每次 reset 会按照抽到的严重度写入相应轮胎表。

当前 `generate_burst_force_tables.py --help` 的帮助文本包含未转义的 `%`，会触发 argparse 格式化错误；上面的实际生成命令不依赖 `--help`。

## 8. 查看命令行帮助

```powershell
python -m scripts.main --help
python -m scripts.train_history_encoder --help
python -m scripts.compare_oracle_history --help
python -m src.utils.restore_full_burst_tables --help
```

## 9. 统计实验注意事项

当前项目没有实现 `--seed` 参数，因此不要在命令后添加无效的 `--seed`。如果需要严格的多随机种子实验，应先在入口脚本中加入种子参数，并同时设置 Python、NumPy、PyTorch、CUDA 和环境采样种子。

建议每项正式实验至少保存：

- 唯一的 PPO checkpoint；
- 对应的 reward 代码版本；
- 完整超参数命令；
- TensorBoard 日志；
- 固定且独立的评估工况表；
- 每个工况的轨迹、稳定性、动作、失败与约束指标。

为了避免覆盖，可以按实验命名文件，例如：

```text
checkpoints\ppo_reward_v2_fresh.pth
checkpoints\ppo_reward_v2_finetune.pth
checkpoints\history_reward_v2.pth
datasets\adaptation_reward_v2.pt
```
