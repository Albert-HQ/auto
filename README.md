# Safe MARL in Autonomous Driving

本项目提供两类算法与四个可运行场景：
- 连续动作（CS-MATD3）：Intersection、Racetrack
- 离散动作（MIP + PER）：Merge、Roundabout

下文给出从冒烟测试到正式训练与评估（含视频导出）的完整、可复制执行的流程。

## 0. 安装环境

```powershell
# 建议使用 Windows PowerShell（Linux/Mac 将 powershell 换成 bash 即可）
conda create -n safe-marl python==3.9 -y
conda activate safe-marl
pip install -r requirements.txt
# 可选：安装与你 CUDA 匹配的 GPU 版 PyTorch
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 1. 冒烟测试（Smoke Test）
目的：用极小配置快速验证“能跑通且不报错”。不会生成视频。

说明：
- 使用 `--smoke-test` 开关，脚本会将 time_steps 等缩小（例如 time_steps≈1000、单 seed）。
- 也提供了对应的 `config.json` 示例（可选）。

目录约定（已存在）：
- 连续：`./intersection_env_result/exp2`、`./racetrack_env_result/exp2`
- 离散：`./merge_env_result/exp2`、`./roundabout_env_result/exp2`

### 1.1 连续动作场景（CS-MATD3）命令
强烈注意：连续动作如需启用 CS-MATD3 训练路径，命令行必须带 `--use-cs-matd3`，否则不会进入 `_cs_matd3_update` 更新逻辑（只会走占位逻辑，等同未训练）。

- Intersection（连续）
```powershell
python .\main_bilevel.py ^
  --scenario intersection ^
  --action-type continuous ^
  --version c_bilevel ^
  --use-cs-matd3 ^
  --file-path .\intersection_env_result\exp2 ^
  --smoke-test
```
- Racetrack（连续）
```powershell
python .\main_bilevel.py ^
  --scenario racetrack ^
  --action-type continuous ^
  --version c_bilevel ^
  --use-cs-matd3 ^
  --file-path .\racetrack_env_result\exp2 ^
  --smoke-test
```

Linux/Mac（bash）可将换行与转义替换为 `\` 或一行写全，例如：
```bash
python main_bilevel.py --scenario intersection --action-type continuous --version c_bilevel --use-cs-matd3 --file-path ./intersection_env_result/exp2 --smoke-test
```

### 1.2 离散动作场景（MIP + PER）命令
- Merge（离散）
```powershell
python .\main_bilevel.py ^
  --scenario merge ^
  --action-type discrete ^
  --use-mip ^
  --file-path .\merge_env_result\exp2 ^
  --smoke-test
```
- Roundabout（离散）
```powershell
python .\main_bilevel.py ^
  --scenario roundabout ^
  --action-type discrete ^
  --use-mip ^
  --file-path .\roundabout_env_result\exp2 ^
  --smoke-test
```

### 1.3 冒烟测试的 config.json 示例（可选）
一般无需改动，直接用 `--smoke-test` 即可。如果你希望通过配置文件控制，也可以将目标目录下的 `config.json` 替换为下述内容之一。

- 连续（Intersection 示例，可将 `scenario_name` 改为 `racetrack-v0` 用于 Racetrack）
```json
{
  "scenario_name": "intersection-v0",
  "action_type": "continuous",
  "version": "c_bilevel",
  "use_cs_matd3": true,
  "time_steps": 1000,
  "max_episode_len": 100,
  "batch_size": 64,
  "sample_size": 128,
  "updates_per_step": 1,
  "policy_delay": 2,
  "target_noise_sigma": 0.2,
  "noise_clip": 0.5,
  "expl_sigma": 0.1,
  "gamma": 0.95,
  "tau": 0.01,
  "cost_threshold": 2,
  "enable_cost": false,
  "evaluate": false,
  "evaluate_rate": 1000,
  "evaluate_episodes": 2,
  "record_video": false
}
```

- 离散（Merge 示例，可将 `scenario_name` 改为 `roundabout-v0` 用于 Roundabout）
```json
{
  "scenario_name": "merge-v0",
  "action_type": "discrete",
  "use_mip": true,
  "time_steps": 1000,
  "max_episode_len": 100,
  "batch_size": 64,
  "sample_size": 128,
  "updates_per_step": 1,
  "gamma": 0.95,
  "tau": 0.01,
  "mip_big_m": 1000.0,
  "per_alpha": 0.6,
  "per_beta_start": 0.4,
  "per_beta_increment": 0.0,
  "per_eps": 1e-6,
  "evaluate": false,
  "evaluate_rate": 1000,
  "evaluate_episodes": 2,
  "record_video": false
}
```

## 2. 正式训练（time_steps = 200000）
- 默认会按 `seed = [0,1,2]` 依次训练并保存到 `.../seed_{seed}` 目录。
- 建议先准备好对应目录下的 `config.json`（见 2.2 示例）。

### 2.1 训练命令（四个场景）
- Intersection（连续，CS-MATD3）
```powershell
python .\main_bilevel.py ^
  --scenario intersection ^
  --action-type continuous ^
  --version c_bilevel ^
  --use-cs-matd3 ^
  --file-path .\intersection_env_result\exp2
```
- Racetrack（连续，CS-MATD3）
```powershell
python .\main_bilevel.py ^
  --scenario racetrack ^
  --action-type continuous ^
  --version c_bilevel ^
  --use-cs-matd3 ^
  --file-path .\racetrack_env_result\exp2
```
- Merge（离散，MIP + PER）
```powershell
python .\main_bilevel.py ^
  --scenario merge ^
  --action-type discrete ^
  --use-mip ^
  --file-path .\merge_env_result\exp2
```
- Roundabout（离散，MIP + PER）
```powershell
python .\main_bilevel.py ^
  --scenario roundabout ^
  --action-type discrete ^
  --use-mip ^
  --file-path .\roundabout_env_result\exp2
```

### 2.2 正式训练的 config.json 示例
训练阶段不生成视频，评估阶段再生成视频（见第 3 节）。

- 连续（Intersection 示例，可改 `scenario_name` 为 `racetrack-v0`）
```json
{
  "scenario_name": "intersection-v0",
  "action_type": "continuous",
  "version": "c_bilevel",
  "use_cs_matd3": true,
  "time_steps": 200000,
  "max_episode_len": 300,
  "batch_size": 256,
  "sample_size": 512,
  "updates_per_step": 1,
  "policy_delay": 2,
  "target_noise_sigma": 0.2,
  "noise_clip": 0.5,
  "expl_sigma": 0.1,
  "gamma": 0.95,
  "tau": 0.01,
  "cost_threshold": 2,
  "enable_cost": false,
  "evaluate": false,
  "evaluate_rate": 5000,
  "evaluate_episodes": 10,
  "evaluate_episode_len": 300,
  "record_video": false
}
```

- 离散（Merge 示例，可改 `scenario_name` 为 `roundabout-v0`）
```json
{
  "scenario_name": "merge-v0",
  "action_type": "discrete",
  "use_mip": true,
  "time_steps": 200000,
  "max_episode_len": 300,
  "batch_size": 256,
  "sample_size": 512,
  "updates_per_step": 1,
  "gamma": 0.95,
  "tau": 0.01,
  "mip_big_m": 1000.0,
  "per_alpha": 0.6,
  "per_beta_start": 0.4,
  "per_beta_increment": 0.0,
  "per_eps": 1e-6,
  "evaluate": false,
  "evaluate_rate": 5000,
  "evaluate_episodes": 10,
  "evaluate_episode_len": 300,
  "record_video": false
}
```

## 3. 评估与导出视频
- 评估阶段建议生成视频：将 `record_video` 设为 `true`，并把 `evaluate` 设为 `true`。
- 运行命令仍与训练一致（由各目录下 `config.json` 控制评估与录像）。
- 评估完的视频保存在：`{save_dir}/video.eval.mp4`；若录制训练视频，则为 `{save_dir}/video.train.mp4`。

示例（先将对应目录的 `config.json` 切换为下述“评估配置”）：（大家可以自行更改评估配置，缩短评估时间）

- 连续（Intersection 示例，可改 `scenario_name` 为 `racetrack-v0`）
```json
{
  "scenario_name": "intersection-v0",
  "action_type": "continuous",
  "version": "c_bilevel",
  "use_cs_matd3": true,
  "time_steps": 1000,
  "max_episode_len": 300,
  "batch_size": 256,
  "sample_size": 512,
  "updates_per_step": 1,
  "policy_delay": 2,
  "target_noise_sigma": 0.2,
  "noise_clip": 0.5,
  "expl_sigma": 0.1,
  "gamma": 0.95,
  "tau": 0.01,
  "cost_threshold": 2,
  "enable_cost": false,
  "evaluate": true,
  "evaluate_rate": 5000,
  "evaluate_episodes": 5,
  "evaluate_episode_len": 200,
  "record_video": true,
  "video_seconds": 40
}
```

- 离散（Merge 示例，可改 `scenario_name` 为 `roundabout-v0`）
```json
{
  "scenario_name": "merge-v0",
  "action_type": "discrete",
  "use_mip": true,
  "time_steps": 200000,
  "max_episode_len": 300,
  "batch_size": 256,
  "sample_size": 512,
  "updates_per_step": 1,
  "gamma": 0.95,
  "tau": 0.01,
  "mip_big_m": 1000.0,
  "per_alpha": 0.6,
  "per_beta_start": 0.4,
  "per_beta_increment": 0.0,
  "per_eps": 1e-6,
  "evaluate": true,
  "evaluate_rate": 5000,
  "evaluate_episodes": 10,
  "evaluate_episode_len": 300,
  "record_video": true,
  "video_seconds": 60
}
```

切换为评估配置后，直接复用第 2.1 节的命令。例如（Intersection）：
```powershell
python .\main_bilevel.py ^
  --scenario intersection ^
  --action-type continuous ^
  --version c_bilevel ^
  --use-cs-matd3 ^
  --file-path .\intersection_env_result\exp2
```

- 显式给出四个场景的评估命令（与训练命令相同），确保对应目录的 `config.json` 已设置 `evaluate:true` 和 `record_video:true`：
  - Intersection（连续）
    ```powershell
    python .\main_bilevel.py --scenario intersection --action-type continuous --version c_bilevel --use-cs-matd3 --file-path .\intersection_env_result\exp2
    ```
  - Racetrack（连续）
    ```powershell
    python .\main_bilevel.py --scenario racetrack --action-type continuous --version c_bilevel --use-cs-matd3 --file-path .\racetrack_env_result\exp2
    ```
  - Merge（离散）
    ```powershell
    python .\main_bilevel.py --scenario merge --action-type discrete --use-mip --file-path .\merge_env_result\exp2
    ```
  - Roundabout（离散）
    ```powershell
    python .\main_bilevel.py --scenario roundabout --action-type discrete --use-mip --file-path .\roundabout_env_result\exp2
    ```

## 4. 参数要点（常用）
- 连续相关（CS-MATD3）
  - use_cs_matd3：是否启用 CS-MATD3 路径（必须在命令行加 `--use-cs-matd3` 才会生效）。
  - policy_delay：策略延迟更新步数（默认 2）。
  - target_noise_sigma / noise_clip：目标策略平滑噪声及其裁剪。
  - expl_sigma：连续动作选择时的探索噪声。
- 通用训练
  - time_steps：训练总步数，正式训练建议 200000。
  - batch_size / sample_size：每次更新采样与批量大小（默认 256/512）。
  - gamma / tau：折扣因子和软更新系数。
  - evaluate_rate / evaluate_episodes / evaluate_episode_len：评估频率、评估轮数与单次评估步长。
  - record_video：是否在训练或评估阶段录制视频（评估阶段建议开启）。
- 离散相关（MIP + PER）
  - use_mip：启用 MIP 进行离散动作选择（命令行加 `--use-mip`）。
  - mip_big_m / mip_solver：MIP 约束的大 M 与求解器（默认 gurobi，亦可 pulp）。
  - per_alpha / per_beta_start / per_beta_increment / per_eps：PER 相关超参。

## 5. 注意事项
- 建议通过 `config.json` 切换训练/评估与是否录像；不建议在命令行直接传 `--evaluate/--record-video`（这些布尔参数在 argparse 中易被误用）。
- 训练日志与视频输出位置：`{file_path}/seed_{seed}`（种子默认 [0,1,2]；冒烟测试仅 [0]）。
- Windows/PowerShell 使用 `^` 续行；Linux/Mac 可使用 `\` 续行或写成一行。

## 6. 引用
如果本仓库对你的研究有帮助，请引用：
```bibtex
@article{zheng2024safe,
  title={Safe Multi-Agent Reinforcement Learning with Bilevel Optimization in Autonomous Driving},
  author={Zheng, Zhi and Gu, Shangding},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2024}
}
```



