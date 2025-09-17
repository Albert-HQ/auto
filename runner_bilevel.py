from tqdm import tqdm
from bilevel_maddpg.replay_buffer import Buffer
from bilevel_maddpg.leader_agent import Leader
from bilevel_maddpg.follower_agent import Follower
from bilevel_maddpg.leader_agent_bilevel import Leader_Bilevel
from bilevel_maddpg.follower_agent_bilevel import Follower_Bilevel
from bilevel_maddpg.per_buffer import PERBuffer
from bilevel_maddpg.model import Critic_Discrete, Cost_Discrete
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers import RecordVideo
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import glob  # 新增：用于查找并重命名最新生成的视频文件
# 新增：PER 与 MIP 依赖
import torch.nn.functional as F
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD, LpStatusOptimal
# 可选：Gurobi 适配
try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except Exception:
    _GUROBI_AVAILABLE = False
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print('is_cuda=',torch.cuda.is_available(),' torch=',torch.__version__,' cuda=',torch.version.cuda)

# 统一将环境观测转换为 (2,8) 的两智能体观测（若给出 (N,7) 则截取前两行并零填充到8维）
def _obs_to_agents(obs, n_agents: int = 2, expected_dim: int = 8) -> np.ndarray:
    arr = np.asarray(obs)
    # 已是扁平 16 长度
    flat = arr.reshape(-1)
    if flat.size == n_agents * expected_dim:
        return flat.reshape(n_agents, expected_dim)
    # 常见：形状 (N, 7) 的 Kinematics
    if arr.ndim == 2 and arr.shape[0] >= n_agents and arr.shape[1] == 7:
        s = arr[:n_agents, :]
        pad = np.zeros((n_agents, expected_dim - 7), dtype=s.dtype) if expected_dim > 7 else None
        return np.concatenate([s, pad], axis=1) if pad is not None else s
    # 若是 (n_agents, 7) 的扁平 14
    if flat.size == n_agents * 7:
        s = flat.reshape(n_agents, 7)
        pad = np.zeros((n_agents, expected_dim - 7), dtype=s.dtype) if expected_dim > 7 else None
        return np.concatenate([s, pad], axis=1) if pad is not None else s
    # 兜底：尽力取前 expected_dim 并广播
    try:
        if arr.ndim == 1 and arr.size >= expected_dim * n_agents:
            return arr[:expected_dim * n_agents].reshape(n_agents, expected_dim)
    except Exception:
        pass
    raise ValueError(f"Unsupported observation shape {arr.shape}, cannot convert to ({n_agents},{expected_dim}).")

# implemntation of Constrained Bilevel RL algorithm
class Runner_C_Bilevel:
    def __init__(self, args, env, eval_env=None):
        self.args = args
        # init noise and epsilon
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        # set min noise and epsilon
        self.min_noise = args.min_noise_rate
        self.min_epsilon = args.min_epsilon
        # set max episode len
        self.episode_limit = args.max_episode_len
        # train env and eval env
        self.env = env
        self.eval_env = eval_env
        # replay buffer
        self.buffer = Buffer(args)
        # save path for training results
        self.save_path = self.args.save_dir
        # reward or cost record
        self.reward_record = [[] for i in range(args.n_agents)]
        self.arrive_record = []
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        # 连续 CS-MATD3 相关超参（提供默认值，避免与离散流程冲突）
        self.use_cs_matd3 = getattr(args, 'use_cs_matd3', True)
        self.updates_per_step = getattr(args, 'updates_per_step', 1)  # K
        self.policy_delay = getattr(args, 'policy_delay', 2)          # 延迟策略更新 d
        self.target_noise_sigma = getattr(args, 'target_noise_sigma', 0.2)  # 目标策略平滑噪声 σ
        self.noise_clip = getattr(args, 'noise_clip', 0.5)                   # 噪声裁剪 c
        self.expl_sigma = getattr(args, 'expl_sigma', 0.1)                   # 动作探索噪声
        # 成本阈值（可单独配置每个agent）
        self.d1 = getattr(args, 'cost_threshold_leader', getattr(args, 'cost_threshold', 0.0))
        self.d2 = getattr(args, 'cost_threshold_follower', getattr(args, 'cost_threshold', 0.0))
        # 拉格朗日乘子学习率
        self.lambda_lr = getattr(args, 'lr_lagrangian', 1e-3)
        # 初始化乘子记录
        self.lambda_leader_hist = []
        self.lambda_follower_hist = []
        self.cost_violation_leader = []
        self.cost_violation_follower = []
        # 日志：拉格朗日乘子与约束违反
        self.lambda_log = []  # [(step, lambda_leader, lambda_follower)]
        self.cost_violation_log = []  # [(step, cv_leader, cv_follower)]
        # agent 初始化
        self._init_agents()
        # 设备引用（两agent应一致）
        self.device = getattr(self.leader_agent, 'device', torch.device('cpu'))
        # make dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # 模型保存频率
        self.model_save_rate = getattr(args, 'model_save_rate', getattr(args, 'evaluate_rate', 5000))

    def _init_agents(self):
        # constrained agents (with cost & multipliers)
        self.leader_agent = Leader(self.args, 0)
        self.follower_agent = Follower(self.args, 1)
        # attach individual thresholds if provided
        self.leader_agent.cost_threshold = float(self.d1)
        self.follower_agent.cost_threshold = float(self.d2)

    # === 新增：统一的模型保存 / 加载接口 ===
    def save_models(self, step='final'):
        try:
            self.leader_agent.save_model(step)
            self.follower_agent.save_model(step)
        except Exception as e:
            print('[WARN][Runner_C_Bilevel] save_models failed:', e)

    def load_models(self):
        # Leader/Follower 在 __init__ 已尝试自动载入；此处保留兼容接口
        # 可在未来扩展自定义文件名规则
        pass

    # 计算给定秒数对应的环境步数（基于 sim_hz 与默认 2x 视频帧率）
    def _seconds_to_steps(self, env_like, seconds: int) -> int:
        sim_hz = getattr(env_like.config, 'simulation_frequency', None)
        if sim_hz is None:
            sim_hz = env_like.config.get('simulation_frequency', 15)
        pol_hz = env_like.config.get('policy_frequency', 5)
        step_time = 1.0 / max(1, pol_hz)
        return int(seconds / step_time)

    # 辅助：录制给定秒数的视频并重命名（仅一个文件）
    def _record_seconds(self, base_env, seconds: int, out_basename: str):
        # 关闭已有viewer，避免残留
        try:
            if getattr(base_env, 'viewer', None) is not None:
                base_env.close()
        except Exception:
            pass
        # 预先计算固定时长步数
        steps = self._seconds_to_steps(base_env, seconds)
        prefix = f"tmp_{out_basename}"
        # 仅记录第 0 个 episode，固定长度，避免碎片
        rec_env = RecordVideo(
            base_env,
            video_folder=self.args.save_dir,
            name_prefix=prefix,
            episode_trigger=lambda eid: eid == 0,
            video_length=steps,
        )
        if hasattr(base_env, 'set_record_video_wrapper'):
            base_env.set_record_video_wrapper(rec_env)
            # 设置视频以实时速率计数
            try:
                base_env.update_metadata(video_real_time_ratio=1)
            except Exception:
                pass
        # 无头渲染录制
        s, info = rec_env.reset(options={"config": {
            "offscreen_rendering": True,
            "render_agent": False,
            "real_time_rendering": False,
            "manual_control": False
        }})
        s = _obs_to_agents(s)
        for _ in range(steps):
            # 不调用 rec_env.render()，避免 gym 版本差异
            with torch.no_grad():
                # 用确定性策略录制评估画面质量
                leader_action = self.leader_agent.select_action(s[0], 0.0, 0.0)
                follower_action = self.follower_agent.select_action(s[1], leader_action, 0.0, 0.0)
            actions = (leader_action, follower_action)
            s_next, r, done, truncated_n, info = rec_env.step(actions)
            s_next = _obs_to_agents(s_next)
            s = s_next
            if np.all(done):
                # 继续录制到固定长度（不会再次触发新文件）
                s, info = rec_env.reset()
                s = _obs_to_agents(s)
        rec_env.close()
        # 将最新生成的视频重命名为所需文件名
        try:
            candidates = sorted(glob.glob(os.path.join(self.args.save_dir, f"{prefix}*.mp4")), key=os.path.getmtime)
            if candidates:
                target = os.path.join(self.args.save_dir, f"{out_basename}.mp4")
                if os.path.exists(target):
                    os.remove(target)
                os.replace(candidates[-1], target)
        except Exception:
            pass

    def run(self):
        returns = []
        total_reward = [0, 0]
        done = [False]*self.args.n_agents  # 修复初始化
        info = None
        render_train = bool(getattr(self.args, 'render_during_train', False))
        # 新增：每集步计数器
        ep_step = 0
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step==0 or np.all(done):
                # save episode rewards
                for i in range(self.args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                # save episode end stats (crash + first_arrived)
                if info is not None:
                    # 统一记录 crash（对所有场景生效）
                    crashed = int(info.get("crash", info.get("crashed", 0)))
                    self.crash_record.append(crashed)
                    # 保留先到达者统计
                    if info.get("first_arrived", 0)==1:
                        self.leader_arrive_record.append(1)
                        self.follower_arrive_record.append(0)
                    elif info.get("first_arrived", 0)==2:
                        self.leader_arrive_record.append(0)
                        self.follower_arrive_record.append(1)
                    else:
                        self.leader_arrive_record.append(0)
                        self.follower_arrive_record.append(0)
                # reset episode total reward
                total_reward = [0, 0]
                # reset（开启可视化显示）
                s, info = self.env.reset(options={"config": {
                    "offscreen_rendering": not render_train,
                    "render_agent": render_train,
                    "real_time_rendering": render_train,
                    "manual_control": False
                }})
                # reshape observation
                s = _obs_to_agents(s)
                # 新增：重置每集步数
                ep_step = 0
            with torch.no_grad():
                # choose actions（连续空间：高斯探索；可回退到原有接口）
                if self.use_cs_matd3 and hasattr(self, '_select_actions_continuous'):
                    leader_action, follower_action = self._select_actions_continuous(s)
                else:
                    leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)

            u = [leader_action, follower_action]
            actions = (leader_action, follower_action)
            # 渲染可视化（默认关闭以加速训练）
            if render_train:
                self.env.render()
            # step simulation（改：接收 env 的 done 与 truncated）
            step_ret = self.env.step(actions)
            try:
                # gym>=0.26: 5 元组 (obs, reward, terminated_or_done, truncated, info)
                if isinstance(step_ret, tuple) and len(step_ret) == 5:
                    s_next, r, done_env, truncated_n, info = step_ret
                # gym<=0.25: 4 元组 (obs, reward, done, info)
                elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                    s_next, r, done_env, info = step_ret
                    truncated_n = False
                else:
                    # 兜底：按 4 元组处理
                    s_next, r, done_env, info = step_ret
                    truncated_n = False
            except Exception:
                # 极端兜底
                s_next, r, done_env, info = step_ret[0], step_ret[1], step_ret[2], step_ret[-1]
                truncated_n = False
            # reshape observation
            s_next = _obs_to_agents(s_next)
            # 新增：集内步数 +1，并在达到上限时强制截断本集
            ep_step += 1
            # 合并 done 与 truncated
            try:
                done_arr = np.array(done_env, dtype=bool)
            except Exception:
                done_arr = np.array([bool(done_env)]*self.args.n_agents)
            if done_arr.ndim == 0:
                done_arr = np.array([bool(done_arr)]*self.args.n_agents)
            # 若 env 提供 truncated，合并到 done（某些 gym 版本返回向量，某些返回标量）
            try:
                trunc_arr = np.array(truncated_n, dtype=bool)
                if trunc_arr.ndim == 0:
                    trunc_arr = np.array([bool(trunc_arr)]*self.args.n_agents)
                done_arr = np.logical_or(done_arr, trunc_arr)
            except Exception:
                pass
            # time-limit 截断：达到 max_episode_len 时强制所有 agent 结束
            if ep_step >= int(self.episode_limit):
                # 在 info 中标注，便于后续分析
                try:
                    info = dict(info) if info is not None else {}
                    info['truncated_episode'] = True
                    info['episode_step'] = int(ep_step)
                except Exception:
                    pass
                done_arr[:] = True
            done = done_arr.tolist()
            # cost
            c = info.get("cost", [0.0, 0.0])
            # next actions(place holder, not used)
            u_next = [0, 0]
            # store transitions（使用合成后的 done，保证 time-limit 被写入）
            done_targets = [np.array([float(done_arr[i])], dtype=np.float32) for i in range(self.args.n_agents)]
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], u_next, done_targets, c=c)
            # observe update
            s = s_next
            # accumulate episode reward
            for i in range(self.args.n_agents):
                total_reward[i]+=r[i]
            # train
            if self.buffer.current_size >= self.args.sample_size:
                if self.use_cs_matd3 and hasattr(self, '_cs_matd3_update'):
                    for k in range(self.updates_per_step):
                        self._cs_matd3_update(time_step)
                else:
                    transitions = self.buffer.sample(self.args.batch_size)
                    # 旧接口保留（若需要可实现）
                    if hasattr(self.leader_agent, 'train') and self.leader_agent.train is not None:
                        pass
            # plot reward / evaluate / periodic save
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                np.save(self.save_path + '/reward_record.npy', self.reward_record)
                np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
                np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
                np.save(self.save_path + '/crash_record.npy', self.crash_record)
                # 额外保存乘子与约束违背
                np.save(self.save_path + '/lambda_leader.npy', np.array(self.lambda_leader_hist, dtype=np.float32))
                np.save(self.save_path + '/lambda_follower.npy', np.array(self.lambda_follower_hist, dtype=np.float32))
                np.save(self.save_path + '/cost_violation_leader.npy', np.array(self.cost_violation_leader, dtype=np.float32))
                np.save(self.save_path + '/cost_violation_follower.npy', np.array(self.cost_violation_follower, dtype=np.float32))
                self.evaluate()
            if self.model_save_rate and (time_step+1) % self.model_save_rate == 0:
                self.save_models(step=time_step+1)
            self.noise = max(self.min_noise, self.noise - 0.0000005)
            self.epsilon = max(self.min_epsilon, self.epsilon - 0.0000005)
            
        # save data (final)
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        np.save(self.save_path + '/leader_arrive_record.npy', self.leader_arrive_record)
        np.save(self.save_path + '/follower_arrive_record.npy', self.follower_arrive_record)
        np.save(self.save_path + '/crash_record.npy', self.crash_record)
        # 额外保存乘子与约束违反日志
        if self.lambda_log:
            np.save(self.save_path + '/lambda_log.npy', np.array(self.lambda_log, dtype=np.float32))
        if self.cost_violation_log:
            np.save(self.save_path + '/cost_violation_log.npy', np.array(self.cost_violation_log, dtype=np.float32))
        # 保存模型
        self.save_models(step='final')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # 根据是否是显式评估，决定是否开启可视化与实时渲染
            show = bool(getattr(self.args, 'evaluate', False))
            s, info = self.eval_env.reset(options={"config": {
                "offscreen_rendering": (not show),
                "render_agent": show,
                "real_time_rendering": show,
                "manual_control": False
            }})
            s = _obs_to_agents(s)
            rewards = [0, 0]
            for time_step in range(self.args.evaluate_episode_len):
                if show:
                    self.eval_env.render()
                with torch.no_grad():
                    # Deterministic evaluation: no noise, no epsilon exploration
                    leader_action = self.leader_agent.select_action(s[0], 0.0, 0.0)
                    follower_action = self.follower_agent.select_action(s[1], leader_action, 0.0, 0.0)
                actions = tuple([leader_action, follower_action])
                s_next, r, done, truncated_n, info = self.eval_env.step(actions)
                s_next = _obs_to_agents(s_next)
                rewards[0] += r[0]
                rewards[1] += r[1]
                s = s_next
                if np.all(done):
                    # 非可视化（训练期间的周期评估）直接结束；显式评估则重置以播放完整时长
                    if show:
                        s, info = self.eval_env.reset(options={"config": {
                            "offscreen_rendering": False,
                            "render_agent": True,
                            "real_time_rendering": True,
                            "manual_control": False
                        }})
                        s = _obs_to_agents(s)
                    else:
                        break
            returns.append(rewards)
            print('Returns is', rewards)
        return np.sum(returns, axis=0) / self.args.evaluate_episodes
    
    def _record_seconds_to_steps(self, base_env, seconds: int) -> int:
        # 在使用 RecordVideo 的情况下，fps = 2 * simulation_frequency（见 AbstractEnv.update_metadata）
        fps = base_env.metadata.get('video.frames_per_second', 30)
        sim_hz = base_env.config.get('simulation_frequency', 15)
        ratio = max(1, int(round(fps / max(1, sim_hz))))  # 即 video_real_time_ratio，通常为2
        return int(seconds * ratio)

    def _rename_last_video(self, folder: str, pattern: str, target_name: str):
        files = sorted(glob.glob(os.path.join(folder, f"{pattern}*.mp4")), key=os.path.getmtime)
        if not files:
            return
        src = files[-1]
        dst = os.path.join(folder, target_name)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.replace(src, dst)
        except Exception:
            pass
    
    def record_video_eval(self):
        # 强制无头渲染，并用 RecordVideo 捕获固定长度的单个视频
        try:
            if getattr(self.eval_env, 'viewer', None) is not None:
                self.eval_env.close()
        except Exception:
            pass
        steps = self._seconds_to_steps(self.eval_env, int(getattr(self.args, 'video_seconds', 60)))
        rec_env = RecordVideo(
            self.eval_env,
            video_folder=self.args.save_dir,
            name_prefix='video_eval',
            episode_trigger=lambda eid: eid == 0,
            video_length=steps,
        )
        if hasattr(self.eval_env, 'set_record_video_wrapper'):
            self.eval_env.set_record_video_wrapper(rec_env)
            try:
                self.eval_env.update_metadata(video_real_time_ratio=1)
            except Exception:
                pass
        s, info = rec_env.reset(options={"config": {
            "offscreen_rendering": True,
            "render_agent": False,
            "real_time_rendering": False,
            "manual_control": False
        }})
        s = _obs_to_agents(s)
        for _ in range(steps):
            with torch.no_grad():
                leader_action = self.leader_agent.select_action(s[0], 0, 0)
                follower_action = self.follower_agent.select_action(s[1], leader_action, 0, 0)
            actions = (leader_action, follower_action)
            s_next, r, done, truncated_n, info = rec_env.step(actions)
            s_next = _obs_to_agents(s_next)
            s = s_next
            if np.all(done):
                s, info = rec_env.reset()
                s = _obs_to_agents(s)
        rec_env.close()
        # 重命名为 video.eval.mp4
        self._rename_last_video(self.args.save_dir, 'video_eval', 'video.eval.mp4')

    def record_video_train(self):
        # 强制无头渲染，并用 RecordVideo 在固定步数内录制单个视频
        try:
            if getattr(self.env, 'viewer', None) is not None:
                self.env.close()
        except Exception:
            pass
        steps = self._seconds_to_steps(self.env, int(getattr(self.args, 'video_seconds', 60)))
        rec_env = RecordVideo(
            self.env,
            video_folder=self.args.save_dir,
            name_prefix='video_train',
            episode_trigger=lambda eid: eid == 0,
            video_length=steps,
        )
        if hasattr(self.env, 'set_record_video_wrapper'):
            self.env.set_record_video_wrapper(rec_env)
        s, info = rec_env.reset(options={"config": {
            "offscreen_rendering": True,
            "render_agent": False,
            "real_time_rendering": False,
            "manual_control": False
        }})
        s = _obs_to_agents(s)
        for _ in range(steps):
            with torch.no_grad():
                leader_action = self.leader_agent.select_action(s[0], self.noise, self.epsilon)
                follower_action = self.follower_agent.select_action(s[1], leader_action, self.noise, self.epsilon)
            actions = (leader_action, follower_action)
            s_next, r, done, truncated_n, info = rec_env.step(actions)
            s_next = _obs_to_agents(s_next)
            s = s_next
            if np.all(done):
                s, info = rec_env.reset()
                s = _obs_to_agents(s)
        rec_env.close()
        # 重命名为 video.train.mp4
        self._rename_last_video(self.args.save_dir, 'video_train', 'video.train.mp4')

    # 保留旧接口，指向 eval 录制，避免外部调用出错
    def record_video(self):
        return self.record_video_eval()

    # === CS-MATD3: 连续动作选择（高斯探索） ===
    def _select_actions_continuous(self, s_np: np.ndarray):
        a_high = self.args.high_action
        # leader
        o0 = torch.tensor(s_np[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a0 = self.leader_agent.actor_network(o0).cpu().numpy()
        a0 = a0 + np.random.normal(0.0, self.expl_sigma * a_high, size=a0.shape)
        a0 = np.clip(a0, -a_high, a_high)
        # follower depends on leader action
        o1 = torch.tensor(s_np[1], dtype=torch.float32, device=self.device)
        a0_t = torch.tensor(a0, dtype=torch.float32, device=self.device)
        o1_in = torch.cat([o1, a0_t], dim=0)
        with torch.no_grad():
            a1 = self.follower_agent.actor_network(o1_in).cpu().numpy()
        a1 = a1 + np.random.normal(0.0, self.expl_sigma * a_high, size=a1.shape)
        a1 = np.clip(a1, -a_high, a_high)
        return a0.copy(), a1.copy()

    # === CS-MATD3: 训练更新（双Q、目标平滑、延迟策略更新、乘子更新） ===
    def _cs_matd3_update(self, global_step: int):
        args = self.args
        device = self.device
        batch = self.buffer.sample(args.batch_size)
        gamma = args.gamma
        a_high = args.high_action
        # 整理 batch 张量
        o0 = torch.tensor(batch['o_0'], dtype=torch.float32, device=device)
        o1 = torch.tensor(batch['o_1'], dtype=torch.float32, device=device)
        u0 = torch.tensor(batch['u_0'], dtype=torch.float32, device=device)
        u1 = torch.tensor(batch['u_1'], dtype=torch.float32, device=device)
        r0 = torch.tensor(batch['r_0'], dtype=torch.float32, device=device).view(-1, 1)
        r1 = torch.tensor(batch['r_1'], dtype=torch.float32, device=device).view(-1, 1)
        c0 = torch.tensor(batch['c_0'], dtype=torch.float32, device=device).view(-1, 1)
        c1 = torch.tensor(batch['c_1'], dtype=torch.float32, device=device).view(-1, 1)
        on0 = torch.tensor(batch['o_next_0'], dtype=torch.float32, device=device)
        on1 = torch.tensor(batch['o_next_1'], dtype=torch.float32, device=device)
        d0 = torch.tensor(batch['t_0'], dtype=torch.float32, device=device)
        d1 = torch.tensor(batch['t_1'], dtype=torch.float32, device=device)
        if d0.dim() == 1:
            d0 = d0.view(-1, 1)
        if d1.dim() == 1:
            d1 = d1.view(-1, 1)
        obs_cat = torch.cat([o0, o1], dim=1)
        obs_cat_next = torch.cat([on0, on1], dim=1)

        # === 目标策略平滑 ===
        with torch.no_grad():
            a0_next = self.leader_agent.actor_target_network(on0)
            noise0 = torch.clamp(
                torch.randn_like(a0_next) * self.target_noise_sigma * a_high,
                -self.noise_clip * a_high, self.noise_clip * a_high
            )
            a0_next = torch.clamp(a0_next + noise0, -a_high, a_high)
            o1_in_next = torch.cat([on1, a0_next], dim=1)
            a1_next = self.follower_agent.actor_target_network(o1_in_next)
            noise1 = torch.clamp(
                torch.randn_like(a1_next) * self.target_noise_sigma * a_high,
                -self.noise_clip * a_high, self.noise_clip * a_high
            )
            a1_next = torch.clamp(a1_next + noise1, -a_high, a_high)
            q0n_1 = self.leader_agent.critic_target_network(obs_cat_next, [a0_next, a1_next])
            q0n_2 = self.leader_agent.critic_target_network2(obs_cat_next, [a0_next, a1_next])
            q0n = torch.min(q0n_1, q0n_2)
            y0 = r0 + (1 - d0) * gamma * q0n

            q1n_1 = self.follower_agent.critic_target_network(obs_cat_next, [a0_next, a1_next])
            q1n_2 = self.follower_agent.critic_target_network2(obs_cat_next, [a0_next, a1_next])
            q1n = torch.min(q1n_1, q1n_2)
            y1 = r1 + (1 - d1) * gamma * q1n

            g0n = self.leader_agent.cost_target_network(obs_cat_next, [a0_next, a1_next])
            z0 = c0 + (1 - d0) * gamma * g0n
            g1n = self.follower_agent.cost_target_network(obs_cat_next, [a0_next, a1_next])
            z1 = c1 + (1 - d1) * gamma * g1n

        # === 更新双 Q 与成本网络 ===
        q0_1 = self.leader_agent.critic_network(obs_cat, [u0, u1])
        q0_2 = self.leader_agent.critic_network2(obs_cat, [u0, u1])
        loss_q0_1 = F.mse_loss(q0_1, y0)
        loss_q0_2 = F.mse_loss(q0_2, y0)
        self.leader_agent.critic_optim.zero_grad(); loss_q0_1.backward(); self.leader_agent.critic_optim.step()
        self.leader_agent.critic2_optim.zero_grad(); loss_q0_2.backward(); self.leader_agent.critic2_optim.step()

        q1_1 = self.follower_agent.critic_network(obs_cat, [u0, u1])
        q1_2 = self.follower_agent.critic_network2(obs_cat, [u0, u1])
        loss_q1_1 = F.mse_loss(q1_1, y1)
        loss_q1_2 = F.mse_loss(q1_2, y1)
        self.follower_agent.critic_optim.zero_grad(); loss_q1_1.backward(); self.follower_agent.critic_optim.step()
        self.follower_agent.critic2_optim.zero_grad(); loss_q1_2.backward(); self.follower_agent.critic2_optim.step()

        g0 = self.leader_agent.cost_network(obs_cat, [u0, u1])
        g1 = self.follower_agent.cost_network(obs_cat, [u0, u1])
        loss_g0 = F.mse_loss(g0, z0)
        loss_g1 = F.mse_loss(g1, z1)
        self.leader_agent.cost_optim.zero_grad(); loss_g0.backward(); self.leader_agent.cost_optim.step()
        self.follower_agent.cost_optim.zero_grad(); loss_g1.backward(); self.follower_agent.cost_optim.step()

        if (global_step % self.policy_delay) == 0:
            a0_pi = self.leader_agent.actor_network(o0)
            follower_requires_grad = [p.requires_grad for p in self.follower_agent.actor_network.parameters()]
            for param, req in zip(self.follower_agent.actor_network.parameters(), follower_requires_grad):
                param.requires_grad_(False)
            o1_br = torch.cat([o1, a0_pi], dim=1)
            a1_br = self.follower_agent.actor_network(o1_br)
            for param, req in zip(self.follower_agent.actor_network.parameters(), follower_requires_grad):
                param.requires_grad_(req)

            gap_leader = self.leader_agent.cost_network(obs_cat, [a0_pi, a1_br]) - self.d1
            leader_actor_loss = (-self.leader_agent.critic_network(obs_cat, [a0_pi, a1_br]) + self.leader_agent.l_multiplier * gap_leader).mean()
            self.leader_agent.actor_optim.zero_grad(); leader_actor_loss.backward(); self.leader_agent.actor_optim.step()

            a0_det = a0_pi.detach()
            o1_in_pi = torch.cat([o1, a0_det], dim=1)
            a1_pi = self.follower_agent.actor_network(o1_in_pi)
            gap_follower = self.follower_agent.cost_network(obs_cat, [a0_det, a1_pi]) - self.d2
            follower_actor_loss = (-self.follower_agent.critic_network(obs_cat, [a0_det, a1_pi]) + self.follower_agent.l_multiplier * gap_follower).mean()
            self.follower_agent.actor_optim.zero_grad(); follower_actor_loss.backward(); self.follower_agent.actor_optim.step()

            self.leader_agent._soft_update_target_network()
            self.follower_agent._soft_update_target_network()

            with torch.no_grad():
                gap0 = float(gap_leader.mean().item())
                gap1 = float(gap_follower.mean().item())
                viol0 = max(gap0, 0.0)
                viol1 = max(gap1, 0.0)
            self.leader_agent.l_multiplier = float(np.clip(self.leader_agent.l_multiplier + self.lambda_lr * gap0, 0.0, self.args.lagrangian_max_bound))
            self.follower_agent.l_multiplier = float(np.clip(self.follower_agent.l_multiplier + self.lambda_lr * gap1, 0.0, self.args.lagrangian_max_bound))
            self.lambda_log.append([global_step, self.leader_agent.l_multiplier, self.follower_agent.l_multiplier])
            self.cost_violation_log.append([global_step, viol0, viol1])
            self.lambda_leader_hist.append(self.leader_agent.l_multiplier)
            self.lambda_follower_hist.append(self.follower_agent.l_multiplier)
            self.cost_violation_leader.append(viol0)
            self.cost_violation_follower.append(viol1)

class Runner_Bilevel:
    # 无约束基线的实现：标准MADDPG式更新（单Q，无成本）
    def __init__(self, args, env, eval_env=None):
        self.args = args
        self.env = env
        self.eval_env = eval_env
        self.buffer = Buffer(args)
        self.leader = Leader_Bilevel(args, 0)
        self.follower = Follower_Bilevel(args, 1)
        self.device = getattr(self.leader, 'device', torch.device('cpu'))
        self.noise = args.noise_rate
        self.min_noise = args.min_noise_rate
        self.gamma = args.gamma
        self.save_path = args.save_dir
        self.episode_limit = args.max_episode_len
        self.reward_record = [[] for _ in range(args.n_agents)]
        self.model_save_rate = getattr(args, 'model_save_rate', getattr(args, 'evaluate_rate', 5000))

    def save_models(self, step='final'):
        try:
            self.leader.save_model(step)
            self.follower.save_model(step)
        except Exception as e:
            print('[WARN][Runner_Bilevel] save_models failed:', e)

    def load_models(self):
        # 已在 agent 构造时尝试载入
        pass

    def run(self):
        total_reward = [0.0, 0.0]
        done = [False, False]
        s, info = self.env.reset()
        s = _obs_to_agents(s)
        for t in tqdm(range(self.args.time_steps)):
            with torch.no_grad():
                a0 = self.leader.select_action(s[0], self.noise, 0.0)
                o1_t = torch.tensor(np.concatenate([s[1], a0], axis=0), dtype=torch.float32, device=self.device)
                a1 = self.follower.actor_network(o1_t).cpu().numpy()
                a1 = np.clip(a1 + np.random.normal(0, self.noise*self.args.high_action, size=a1.shape), -self.args.high_action, self.args.high_action)
            actions = (a0, a1)
            # 修复：缺失的 env.step 赋值语句
            s_next, r, done, truncated, info = self.env.step(actions)
            s_next = _obs_to_agents(s_next)
            done_arr = np.array(done, dtype=np.float32)
            if done_arr.ndim == 0:
                done_arr = np.repeat(done_arr, self.args.n_agents)
            done_targets = [np.array([float(done_arr[i])], dtype=np.float32) for i in range(self.args.n_agents)]
            self.buffer.store_episode(s, [a0, a1], r, s_next, [0, 0], done_targets, c=[0.0, 0.0])
            total_reward[0]+=r[0]; total_reward[1]+=r[1]
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                batch = self.buffer.sample(self.args.batch_size)
                self._update(batch)
            done = done_arr.astype(bool)
            if np.all(done):
                for i in range(2):
                    self.reward_record[i].append(total_reward[i])
                total_reward = [0.0, 0.0]
                s, info = self.env.reset()
                s = _obs_to_agents(s)
            if self.model_save_rate and (t+1) % self.model_save_rate == 0:
                self.save_models(step=t+1)
        # final save
        self.save_models(step='final')

    def _update(self, batch):
        # tensors
        o0 = torch.tensor(batch['o_0'], dtype=torch.float32, device=self.device)
        o1 = torch.tensor(batch['o_1'], dtype=torch.float32, device=self.device)
        obs_cat = torch.cat([o0, o1], dim=1)
        u0 = torch.tensor(batch['u_0'], dtype=torch.float32, device=self.device)
        u1 = torch.tensor(batch['u_1'], dtype=torch.float32, device=self.device)
        r0 = torch.tensor(batch['r_0'], dtype=torch.float32, device=self.device).view(-1,1)
        r1 = torch.tensor(batch['r_1'], dtype=torch.float32, device=self.device).view(-1,1)
        on0 = torch.tensor(batch['o_next_0'], dtype=torch.float32, device=self.device)
        on1 = torch.tensor(batch['o_next_1'], dtype=torch.float32, device=self.device)
        obs_cat_next = torch.cat([on0, on1], dim=1)
        d0 = torch.tensor(batch['t_0'], dtype=torch.float32, device=self.device).view(-1,1)
        d1 = torch.tensor(batch['t_1'], dtype=torch.float32, device=self.device).view(-1,1)
        # target actions
        with torch.no_grad():
            a0n = self.leader.actor_target_network(on0)
            o1_in = torch.cat([on1, a0n], dim=1)
            a1n = self.follower.actor_target_network(o1_in)
            q0_target = r0 + (1-d0)*self.gamma * self.leader.critic_target_network(obs_cat_next, [a0n, a1n])
            q1_target = r1 + (1-d1)*self.gamma * self.follower.critic_target_network(obs_cat_next, [a0n, a1n])
        # critic update
        q0 = self.leader.critic_network(obs_cat, [u0, u1])
        q1 = self.follower.critic_network(obs_cat, [u0, u1])
        loss0 = F.mse_loss(q0, q0_target)
        loss1 = F.mse_loss(q1, q1_target)
        self.leader.critic_optim.zero_grad(); loss0.backward(); self.leader.critic_optim.step()
        self.follower.critic_optim.zero_grad(); loss1.backward(); self.follower.critic_optim.step()
        # actor update
        a0_pi = self.leader.actor_network(o0)
        o1_in_pi = torch.cat([o1, a0_pi.detach()], dim=1)
        a1_pi = self.follower.actor_network(o1_in_pi)
        actor0_loss = - self.leader.critic_network(obs_cat, [a0_pi, a1_pi]).mean()
        actor1_loss = - self.follower.critic_network(obs_cat, [a0_pi.detach(), a1_pi]).mean()
        self.leader.actor_optim.zero_grad(); actor0_loss.backward(); self.leader.actor_optim.step()
        self.follower.actor_optim.zero_grad(); actor1_loss.backward(); self.follower.actor_optim.step()
        # soft update
        self.leader._soft_update_target_network()
        self.follower._soft_update_target_network()

    def evaluate(self):
        returns = []
        eval_episodes = getattr(self.args, 'evaluate_episodes', 5)
        ep_len = getattr(self.args, 'evaluate_episode_len', 200)
        for _ in range(eval_episodes):
            s, info = self.eval_env.reset()
            s = _obs_to_agents(s)
            ep_reward = [0.0 for _ in range(self.args.n_agents)]
            for t in range(ep_len):
                obs_cat = np.concatenate([s[0], s[1]], axis=0)
                q1_np, q2_np, g1_np, g2_np, safe_sets = self._build_tables_and_safe_sets(obs_cat)
                a0, a1 = self._mip_action_full(q1_np, q2_np, g1_np, g2_np, safe_sets)
                s_next, r, done, trunc, info = self.eval_env.step((a0, a1))
                s_next = _obs_to_agents(s_next)
                for i in range(self.args.n_agents):
                    ep_reward[i] += r[i]
                s = s_next
                if np.all(done):
                    break
            returns.append(ep_reward)
        mean_ret = np.mean(np.array(returns), axis=0)
        return mean_ret

# ===================== 离散约束 Stackelberg (MIP-PCSQ) Runner =====================
class Runner_Stochastic:
    """优先经验回放 + 约束 Stackelberg Q 学习 + MIP 动作选择 (离散)。
    特性:
      - 对齐伪代码 (MIP-PCSQ) 主要步骤 (标注 Step x)
      - 双Q (Q1,Q2) + 双成本 (G1,G2)
      - PER 优先级更新
      - MIP / 可行集贪心 选择 (a1,a2) 满足成本阈值 (Step 5~10)
      - next-state 目标动作同样通过可行集/（可选 MIP）产生 (Step 16~18)
    """
    def __init__(self, args, env, eval_env=None):
        self.args = args
        self.env = env
        self.eval_env = eval_env
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(args, 'use_cuda', True) else 'cpu')
        # PER buffer
        self.buffer = PERBuffer(args)
        # 网络（集中式）：输入 = sum(obs_shape)=16, 动作 one-hot= sum(action_dim)=10
        self.q1 = Critic_Discrete(args).to(self.device)
        self.q2 = Critic_Discrete(args).to(self.device)
        self.g1 = Cost_Discrete(args).to(self.device)
        self.g2 = Cost_Discrete(args).to(self.device)
        self.q1_t = Critic_Discrete(args).to(self.device); self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t = Critic_Discrete(args).to(self.device); self.q2_t.load_state_dict(self.q2.state_dict())
        self.g1_t = Cost_Discrete(args).to(self.device); self.g1_t.load_state_dict(self.g1.state_dict())
        self.g2_t = Cost_Discrete(args).to(self.device); self.g2_t.load_state_dict(self.g2.state_dict())
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=args.lr_critic)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=args.lr_critic)
        self.opt_g1 = torch.optim.Adam(self.g1.parameters(), lr=args.lr_critic)
        self.opt_g2 = torch.optim.Adam(self.g2.parameters(), lr=args.lr_critic)
        self.gamma = args.gamma
        # 约束阈值
        self.d1 = getattr(args, 'cost_threshold_leader', args.cost_threshold)
        self.d2 = getattr(args, 'cost_threshold_follower', args.cost_threshold)
        # === 缺陷修复: 初始化乘子（即便伪代码未显式使用，保留监控成本违反的可扩展接口） ===
        self.lambda_lr = getattr(args, 'lr_lagrangian', 1e-3)
        self.lambda_leader = float(getattr(args, 'lagrangian_multiplier', 0.0))
        self.lambda_follower = float(getattr(args, 'lagrangian_multiplier', 0.0))
        # 记录结构
        self.lambda_leader_hist = []
        self.lambda_follower_hist = []
        self.cost_violation_leader = []
        self.cost_violation_follower = []
        self.lambda_log = []
        self.cost_violation_log = []
        # 动作空间
        self.A1, self.A2 = args.action_dim[0], args.action_dim[1]
        self.joint_oh_tensor = self._precompute_joint_onehot().to(self.device)  # [A1*A2, A1+A2]
        # 训练相关
        self.noise = args.noise_rate
        self.min_noise = args.min_noise_rate
        self.epsilon = getattr(args, 'epsilon', 0.1)
        self.min_epsilon = getattr(args, 'min_epsilon', 0.01)
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.9995)
        self.updates_per_step = getattr(args, 'updates_per_step', 1)
        # === 缺陷修复: reward_record 未初始化导致 run() append 报错 ===
        self.reward_record = [[] for _ in range(args.n_agents)]
        # 新增：episode 上限
        self.episode_limit = getattr(args, 'max_episode_len', 200)
        # 保存路径
        self.save_path = args.save_dir
        os.makedirs(self.save_path, exist_ok=True)
        self.model_dir = os.path.join(self.save_path, 'discrete_model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_save_rate = getattr(args, 'model_save_rate', getattr(args, 'evaluate_rate', 5000))
        # 目标网络使用 min(Q1,Q2) 还是单 Q1，可通过开关微调（伪代码使用单 Qtargi，可配置）
        self.use_min_target = getattr(args, 'use_min_target', True)
        # 运行时检查：Gurobi 请求但不可用 -> 回退 PuLP
        solver_pref = str(getattr(self.args, 'mip_solver', 'gurobi')).lower()
        if solver_pref == 'gurobi' and not _GUROBI_AVAILABLE:
            print('[WARN] gurobi solver requested but gurobipy is not available. Falling back to PuLP/CBC.')
            self.args.mip_solver = 'pulp'
        # 如果是评估模式，尝试载入
        if getattr(self.args, 'evaluate', False):
            self.load_models()

    def save_models(self, step='final'):
        try:
            torch.save(self.q1.state_dict(), os.path.join(self.model_dir, f'q1_{step}.pth'))
            torch.save(self.q2.state_dict(), os.path.join(self.model_dir, f'q2_{step}.pth'))
            torch.save(self.g1.state_dict(), os.path.join(self.model_dir, f'g1_{step}.pth'))
            torch.save(self.g2.state_dict(), os.path.join(self.model_dir, f'g2_{step}.pth'))
            # 同步一份最新权重 (latest)
            torch.save(self.q1.state_dict(), os.path.join(self.model_dir, 'q1_latest.pth'))
            torch.save(self.q2.state_dict(), os.path.join(self.model_dir, 'q2_latest.pth'))
            torch.save(self.g1.state_dict(), os.path.join(self.model_dir, 'g1_latest.pth'))
            torch.save(self.g2.state_dict(), os.path.join(self.model_dir, 'g2_latest.pth'))
        except Exception as e:
            print('[WARN][Runner_Stochastic] save_models failed:', e)

    def load_models(self, tag='latest'):
        try:
            q1f = os.path.join(self.model_dir, f'q1_{tag}.pth') if tag!='latest' else os.path.join(self.model_dir, 'q1_latest.pth')
            if os.path.exists(q1f): self.q1.load_state_dict(torch.load(q1f, map_location=self.device))
            q2f = os.path.join(self.model_dir, f'q2_{tag}.pth') if tag!='latest' else os.path.join(self.model_dir, 'q2_latest.pth')
            if os.path.exists(q2f): self.q2.load_state_dict(torch.load(q2f, map_location=self.device))
            g1f = os.path.join(self.model_dir, f'g1_{tag}.pth') if tag!='latest' else os.path.join(self.model_dir, 'g1_latest.pth')
            if os.path.exists(g1f): self.g1.load_state_dict(torch.load(g1f, map_location=self.device))
            g2f = os.path.join(self.model_dir, f'g2_{tag}.pth') if tag!='latest' else os.path.join(self.model_dir, 'g2_latest.pth')
            if os.path.exists(g2f): self.g2.load_state_dict(torch.load(g2f, map_location=self.device))
        except Exception as e:
            print('[WARN][Runner_Stochastic] load_models failed:', e)

    def _soft_update(self, target_net, source_net, tau=None):
        """Polyak软更新: target = tau * source + (1 - tau) * target"""
        if tau is None:
            tau = getattr(self.args, 'tau', 0.005)
        with torch.no_grad():
            for p_t, p_s in zip(target_net.parameters(), source_net.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)
        
    # 预计算所有 joint one-hot
    def _precompute_joint_onehot(self):
        joint_list = []
        for a0 in range(self.A1):
            for a1 in range(self.A2):
                oh0 = np.zeros(self.A1, dtype=np.float32); oh0[a0] = 1
                oh1 = np.zeros(self.A2, dtype=np.float32); oh1[a1] = 1
                joint_list.append(np.concatenate([oh0, oh1], axis=0))
        return torch.tensor(np.stack(joint_list, axis=0), dtype=torch.float32)

    # ==== 严格伪代码对齐部分（替换/覆盖先前简化版本） ====
    # Step 5: 生成 Q/G 表以及每个 leader 动作的安全集合 S_i(s)
    def _build_tables_and_safe_sets(self, obs_cat_np):
        obs_t = torch.tensor(obs_cat_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        q1, q2, g1, g2 = self._all_qg(obs_t)  # [A1*A2]
        q1_np = q1.cpu().numpy(); q2_np = q2.cpu().numpy()
        g1_np = g1.cpu().numpy(); g2_np = g2.cpu().numpy()
        safe_sets = []  # List[List[int]] for each leader action i: follower indices j that satisfy成本
        for i in range(self.A1):
            js = []
            base = i * self.A2
            for j in range(self.A2):
                k = base + j
                if g1_np[k] <= self.d1 and g2_np[k] <= self.d2:
                    js.append(j)
            safe_sets.append(js)
        return q1_np, q2_np, g1_np, g2_np, safe_sets

    # Step 6-10: ε-探索 + MIP 解 (13a-13h)
    def _epsilon_action(self, obs_np):
        obs_cat = np.concatenate([obs_np[0], obs_np[1]], axis=0)
        q1_np, q2_np, g1_np, g2_np, safe_sets = self._build_tables_and_safe_sets(obs_cat)
        # 联合安全集合的全部可行 (i,j)
        feasible_pairs = [(i, j) for i in range(self.A1) for j in safe_sets[i]]
        if np.random.rand() < self.epsilon:
            # 若无可行对，随机整个动作空间 (退化情况)
            if len(feasible_pairs) == 0:
                return np.random.randint(self.A1), np.random.randint(self.A2)
            return feasible_pairs[np.random.randint(len(feasible_pairs))]
        # 利用：解 MIP
        return self._mip_action_full(q1_np, q2_np, g1_np, g2_np, safe_sets)

    # 将 (i,j) 转为扁平索引
    def _pair_to_index(self, a0, a1):
        return a0 * self.A2 + a1

    # 完整 MIP (13a-13h) 实现。safe_sets: S_i(s)
    def _mip_action_full(self, q1_np, q2_np, g1_np, g2_np, safe_sets):
        """解 (13a–13h) MIP：优先用 Gurobi；失败/不可用时回退 PuLP；再不行回退安全集合贪心。
        返回 (i,j)。
        """
        M = float(getattr(self.args, 'mip_big_m', 1e3))
        use_mip = getattr(self.args, 'use_mip', True)
        feasible_pairs = [(i, j) for i in range(self.A1) for j in safe_sets[i]]
        # 非MIP或无可行集合时沿用现有回退策略
        if not use_mip or len(feasible_pairs) == 0:
            if len(feasible_pairs) == 0:
                viol = []
                for i in range(self.A1):
                    for j in range(self.A2):
                        k = self._pair_to_index(i, j)
                        v = max(g1_np[k] - self.d1, 0.0) + max(g2_np[k] - self.d2, 0.0)
                        viol.append((v, -q1_np[k], i, j))
                viol.sort()
                return viol[0][2], viol[0][3]
            best = max(feasible_pairs, key=lambda ij: q1_np[self._pair_to_index(ij[0], ij[1])])
            return best

        # 根据用户偏好选择求解器（默认Gurobi）
        solver_choice = getattr(self.args, 'mip_solver', 'gurobi').lower()
        try_gurobi = (solver_choice == 'gurobi')
        # 先尝试 Gurobi
        if try_gurobi and _GUROBI_AVAILABLE:
            try:
                model = gp.Model('MIP_PCQS_Full')
                model.Params.OutputFlag = 0
                if getattr(self.args, 'mip_timeout', None) is not None:
                    model.Params.TimeLimit = float(self.args.mip_timeout)
                if getattr(self.args, 'gurobi_threads', None) is not None:
                    model.Params.Threads = int(self.args.gurobi_threads)
                # 变量
                x = model.addVars(self.A1, vtype=GRB.BINARY, name='x')
                y = model.addVars(self.A1, self.A2, vtype=GRB.BINARY, name='y')
                v = model.addVars(self.A1, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name='v')
                # 目标 (13a)
                obj = gp.quicksum(float(q1_np[self._pair_to_index(i, j)]) * y[i, j]
                                   for i in range(self.A1) for j in range(self.A2))
                model.setObjective(obj, GRB.MAXIMIZE)
                # (13b) 唯一选择
                model.addConstr(gp.quicksum(y[i, j] for i in range(self.A1) for j in range(self.A2)) == 1, name='c13b')
                # (13c) 行一致性
                for i in range(self.A1):
                    model.addConstr(gp.quicksum(y[i, j] for j in range(self.A2)) == x[i], name=f'c13c_{i}')
                # (13d) safe set 下界：x_i=1 => v_i >= Q_iℓ^2
                for i in range(self.A1):
                    for l in safe_sets[i]:
                        k = self._pair_to_index(i, l)
                        model.addConstr(v[i] >= float(q2_np[k]) - M * (1 - x[i]), name=f'c13d_{i}_{l}')
                # (13e)(13f) 选中联动：y_ij=1 => v_i == Q_ij2（用两条不等式表达）
                for i in range(self.A1):
                    for j in range(self.A2):
                        k = self._pair_to_index(i, j)
                        model.addConstr(v[i] >= float(q2_np[k]) - M * (1 - y[i, j]), name=f'c13e_{i}_{j}')
                        model.addConstr(v[i] <= float(q2_np[k]) + M * (1 - y[i, j]), name=f'c13f_{i}_{j}')
                # (13g)(13h) 成本阈值：y_ij=1 => G_ij1<=d1, G_ij2<=d2
                for i in range(self.A1):
                    for j in range(self.A2):
                        k = self._pair_to_index(i, j)
                        model.addConstr(float(g1_np[k]) - float(self.d1) <= M * (1 - y[i, j]), name=f'c13g_{i}_{j}')
                        model.addConstr(float(g2_np[k]) - float(self.d2) <= M * (1 - y[i, j]), name=f'c13h_{i}_{j}')
                # 求解
                model.optimize()
                if model.Status == GRB.OPTIMAL or model.SolCount > 0:
                    # 读解：找到 y_ij=1
                    for i in range(self.A1):
                        for j in range(self.A2):
                            if y[i, j].X > 0.5:
                                return i, j
                # 若非最优或无解则回退
            except Exception:
                pass
        # 回退 PuLP CBC
        try:
            prob = LpProblem('MIP_PCQS_Full', LpMaximize)
            x = {i: LpVariable(f'x_{i}', 0, 1, cat=LpBinary) for i in range(self.A1)}
            y = {(i, j): LpVariable(f'y_{i}_{j}', 0, 1, cat=LpBinary) for i in range(self.A1) for j in range(self.A2)}
            v = {i: LpVariable(f'v_{i}', lowBound=-M, upBound=M) for i in range(self.A1)}
            prob += lpSum([float(q1_np[self._pair_to_index(i, j)]) * y[(i, j)] for i in range(self.A1) for j in range(self.A2)])
            prob += lpSum([y[(i, j)] for i in range(self.A1) for j in range(self.A2)]) == 1
            for i in range(self.A1):
                prob += lpSum([y[(i, j)] for j in range(self.A2)]) == x[i]
            for i in range(self.A1):
                for l in safe_sets[i]:
                    k = self._pair_to_index(i, l)
                    prob += v[i] >= float(q2_np[k]) - M * (1 - x[i])
            for i in range(self.A1):
                for j in range(self.A2):
                    k = self._pair_to_index(i, j)
                    prob += v[i] >= float(q2_np[k]) - M * (1 - y[(i, j)])
                    prob += v[i] <= float(q2_np[k]) + M * (1 - y[(i, j)])
                    prob += (float(g1_np[k]) - float(self.d1)) - M * (1 - y[(i, j)]) <= 0
                    prob += (float(g2_np[k]) - float(self.d2)) - M * (1 - y[(i, j)]) <= 0
            solver = PULP_CBC_CMD(msg=0, timeLimit=getattr(self.args, 'mip_timeout', None))
            prob.solve(solver)
            if prob.status == LpStatusOptimal:
                for i in range(self.A1):
                    for j in range(self.A2):
                        val = y[(i, j)].value()
                        if val is not None and val > 0.5:
                            return i, j
        except Exception:
            pass
        # 最后兜底：安全集合贪心/最小违反
        if len(feasible_pairs) > 0:
            return max(feasible_pairs, key=lambda ij: q1_np[self._pair_to_index(ij[0], ij[1])])
        best = int(np.argmax(q1_np))
        return best // self.A2, best % self.A2

    # Step 16~18: 在 next state 上解 MIP 选取 (a'1,a'2) (不含 ε 探索)
    def _next_state_action_indices(self, obs_cat_next_batch):
        indices = []
        for obs_cat in obs_cat_next_batch:
            q1_np, q2_np, g1_np, g2_np, safe_sets = self._build_tables_and_safe_sets(obs_cat)
            a0, a1 = self._mip_action_full(q1_np, q2_np, g1_np, g2_np, safe_sets)
            indices.append(self._pair_to_index(a0, a1))
        return indices

    # 覆盖更新函数以严格对齐 Step 18 目标动作选取
    def _update(self, global_step:int):
        batch = self.buffer.sample(self.args.batch_size)
        idx = batch['idx']
        # Step 14-15: 采样 + 重要性采样权重 w, 并归一化得到 w_tilde
        w = torch.tensor(batch['weights'], dtype=torch.float32, device=self.device).view(-1,1)
        w_tilde = w / (w.max() + 1e-8)  # 显式体现 Step15 的归一化 (原 PER 已做, 再显式保证)
        done = torch.tensor(batch['done'], dtype=torch.float32, device=self.device).view(-1,1)
        o0 = torch.tensor(batch['o_0'], dtype=torch.float32, device=self.device)
        o1 = torch.tensor(batch['o_1'], dtype=torch.float32, device=self.device)
        on0 = torch.tensor(batch['o_next_0'], dtype=torch.float32, device=self.device)
        on1 = torch.tensor(batch['o_next_1'], dtype=torch.float32, device=self.device)
        obs_cat = torch.cat([o0, o1], dim=1)
        obs_cat_next = torch.cat([on0, on1], dim=1)
        r0 = torch.tensor(batch['r_0'], dtype=torch.float32, device=self.device).view(-1,1)
        r1 = torch.tensor(batch['r_1'], dtype=torch.float32, device=self.device).view(-1,1)
        c0 = torch.tensor(batch['c_0'], dtype=torch.float32, device=self.device).view(-1,1)
        c1 = torch.tensor(batch['c_1'], dtype=torch.float32, device=self.device).view(-1,1)
        u0 = torch.tensor(batch['u_0'], dtype=torch.long, device=self.device)
        u1 = torch.tensor(batch['u_1'], dtype=torch.long, device=self.device)
        oh0 = F.one_hot(u0, num_classes=self.A1)
        oh1 = F.one_hot(u1, num_classes=self.A2)
        a_oh = torch.cat([oh0, oh1], dim=-1).float()
        # Step 19: 当前 Q / G
        q1 = self.q1(obs_cat, a_oh)
        q2 = self.q2(obs_cat, a_oh)
        g1 = self.g1(obs_cat, a_oh)
        g2 = self.g2(obs_cat, a_oh)
        # Step 16~18: 下个状态动作 (MIP)，并用目标网络估计目标值
        with torch.no_grad():
            next_indices = self._next_state_action_indices(obs_cat_next.cpu().numpy())
            J = self.joint_oh_tensor.size(0)
            B = obs_cat_next.size(0)
            obs_rep = obs_cat_next.unsqueeze(1).repeat(1, J, 1).view(-1, obs_cat_next.size(1))
            joint_rep = self.joint_oh_tensor.unsqueeze(0).repeat(B,1,1).view(-1, self.joint_oh_tensor.size(1))
            q1n_all = self.q1_t(obs_rep, joint_rep).view(B, J)
            q2n_all = self.q2_t(obs_rep, joint_rep).view(B, J)
            g1n_all = self.g1_t(obs_rep, joint_rep).view(B, J)
            g2n_all = self.g2_t(obs_rep, joint_rep).view(B, J)
            sel_idx = torch.tensor(next_indices, dtype=torch.long, device=self.device).view(-1,1)
            if self.use_min_target:
                q_target_all = torch.min(q1n_all, q2n_all)  # TD3 风格保守估计
            else:
                q_target_all = q1n_all  # 严格按伪代码只用单一 target critic
            q_target_sel = q_target_all.gather(1, sel_idx)
            g1_sel = g1n_all.gather(1, sel_idx)
            g2_sel = g2n_all.gather(1, sel_idx)
            y0 = r0 + (1 - done) * self.gamma * q_target_sel
            y1 = r1 + (1 - done) * self.gamma * q_target_sel
            z0 = c0 + (1 - done) * self.gamma * g1_sel
            z1 = c1 + (1 - done) * self.gamma * g2_sel
        # Step 20-21: 加权损失（显式使用 w_tilde）
        dQ1 = q1 - y0
        dQ2 = q2 - y1
        dG1 = g1 - z0
        dG2 = g2 - z1
        loss_q1 = (w_tilde * dQ1.pow(2)).mean()
        loss_q2 = (w_tilde * dQ2.pow(2)).mean()
        self.opt_q1.zero_grad(); loss_q1.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); loss_q2.backward(); self.opt_q2.step()
        loss_g1 = (w_tilde * dG1.pow(2)).mean()
        loss_g2 = (w_tilde * dG2.pow(2)).mean()
        self.opt_g1.zero_grad(); loss_g1.backward(); self.opt_g1.step()
        self.opt_g2.zero_grad(); loss_g2.backward(); self.opt_g2.step()
        # Step 22: 优先级更新 (组合残差)
        td_q1 = dQ1.detach().abs().squeeze(-1).cpu().numpy()
        td_q2 = dQ2.detach().abs().squeeze(-1).cpu().numpy()
        td_g1 = dG1.detach().abs().squeeze(-1).cpu().numpy()
        td_g2 = dG2.detach().abs().squeeze(-1).cpu().numpy()
        p = (self.args.per_lambda_q1 * td_q1 +
             self.args.per_lambda_q2 * td_q2 +
             self.args.per_lambda_g1 * td_g1 +
             self.args.per_lambda_g2 * td_g2)
        self.args.priorities_ready = True
        self.buffer.update_priorities(idx, p)
        # Step 23: 软更新
        self._soft_update(self.q1_t, self.q1, self.args.tau)
        self._soft_update(self.q2_t, self.q2, self.args.tau)
        self._soft_update(self.g1_t, self.g1, self.args.tau)
        self._soft_update(self.g2_t, self.g2, self.args.tau)
        # （可选监控）成本违反与乘子日志
        with torch.no_grad():
            cv_leader = F.relu(g1.mean() - self.d1).item()
            cv_follower = F.relu(g2.mean() - self.d2).item()
        self.lambda_leader = float(np.clip(self.lambda_leader + self.lambda_lr * cv_leader, 0.0, self.args.lagrangian_max_bound))
        self.lambda_follower = float(np.clip(self.lambda_follower + self.lambda_lr * cv_follower, 0.0, self.args.lagrangian_max_bound))
        self.lambda_log.append([global_step, self.lambda_leader, self.lambda_follower])
        self.cost_violation_log.append([global_step, cv_leader, cv_follower])
        self.lambda_leader_hist.append(self.lambda_leader)
        self.lambda_follower_hist.append(self.lambda_follower)
        self.cost_violation_leader.append(cv_leader)
        self.cost_violation_follower.append(cv_follower)

    # === 新增: 生成全部 joint Q/G (供 Step5 使用) ===
    def _all_qg(self, obs_t):
        """返回 (q1,q2,g1,g2) 各展平为 [A1*A2]。obs_t: [1, obs_dim]"""
        with torch.no_grad():
            J = self.A1 * self.A2
            obs_rep = obs_t.repeat(J, 1)              # [J, obs_dim]
            joint = self.joint_oh_tensor.to(self.device)  # [J, A1+A2]
            q1 = self.q1(obs_rep, joint).view(J)
            q2 = self.q2(obs_rep, joint).view(J)
            g1 = self.g1(obs_rep, joint).view(J)
            g2 = self.g2(obs_rep, joint).view(J)
        return q1, q2, g1, g2

    # === 训练主循环：严格按伪代码顺序组织 ===
    def run(self):
        args = self.args
        total_reward = [0.0, 0.0]
        done_vec = [False]*args.n_agents
        info = None
        s, info = self.env.reset()
        s = np.array(s).reshape((2, args.obs_shape[0]))  # 假设各 agent 观测同维度
        # 新增：每集步计数器
        ep_step = 0
        for step in tqdm(range(args.time_steps)):
            # Step 5~10: 选动作 (含 ε 探索 & MIP) （若 episode 结束则重置）
            if step==0 or np.all(done_vec):
                for i in range(args.n_agents):
                    self.reward_record[i].append(total_reward[i])
                total_reward = [0.0, 0.0]
                s, info = self.env.reset()
                s = np.array(s).reshape((2, args.obs_shape[0]))
                # 新增：重置每集步数
                ep_step = 0
            a0, a1 = self._epsilon_action(s)
            # 环境执行 (Step 11) + 兼容 4/5 元组
            step_ret = self.env.step((a0, a1))
            try:
                if isinstance(step_ret, tuple) and len(step_ret) == 5:
                    s_next, r, done_env, truncated, info = step_ret
                elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                    s_next, r, done_env, info = step_ret
                    truncated = False
                else:
                    s_next, r, done_env, info = step_ret
                    truncated = False
            except Exception:
                s_next, r, done_env, info = step_ret[0], step_ret[1], step_ret[2], step_ret[-1]
                truncated = False
            s_next = _obs_to_agents(s_next)
            # 新增：集内步数 +1，并在达到上限时强制截断本集
            ep_step += 1
            # 合并 done 与 truncated
            try:
                done_arr = np.array(done_env, dtype=bool)
            except Exception:
                done_arr = np.array([bool(done_env)]*args.n_agents)
            if done_arr.ndim == 0:
                done_arr = np.array([bool(done_arr)]*args.n_agents)
            try:
                trunc_arr = np.array(truncated, dtype=bool)
                if trunc_arr.ndim == 0:
                    trunc_arr = np.array([bool(trunc_arr)]*args.n_agents)
                done_arr = np.logical_or(done_arr, trunc_arr)
            except Exception:
                pass
            # time-limit 截断
            if ep_step >= int(self.episode_limit):
                try:
                    info = dict(info) if info is not None else {}
                    info['truncated_episode'] = True
                    info['episode_step'] = int(ep_step)
                except Exception:
                    pass
                done_arr[:] = True
            done_vec = done_arr.tolist()
            c = info.get('cost', [0.0, 0.0])
            # Step 12: 存储 transition （done 统一整合）
            self.buffer.store_episode([s[0], s[1]], [a0, a1], [r[0], r[1]], [s_next[0], s_next[1]], [0,0], float(np.all(done_vec)), c=c)
            # 累积奖励
            total_reward[0]+=r[0]; total_reward[1]+=r[1]
            # Step 13~23: 更新 (缓冲区足够后，多次 critic/constraint 更新)
            if self.buffer.current_size >= args.sample_size:
                for _ in range(self.updates_per_step):
                    self._update(step)
            # 衰减 ε
            self.epsilon = max(self.min_epsilon, self.epsilon * getattr(args,'epsilon_decay',0.9995))
            s = s_next
            # 周期评估与保存
            if (step+1) % args.evaluate_rate == 0:
                np.save(self.save_path + '/reward_record.npy', self.reward_record)
                np.save(self.save_path + '/lambda_leader.npy', np.array(self.lambda_leader_hist, dtype=np.float32))
                np.save(self.save_path + '/lambda_follower.npy', np.array(self.lambda_follower_hist, dtype=np.float32))
                np.save(self.save_path + '/cost_violation_leader.npy', np.array(self.cost_violation_leader, dtype=np.float32))
                np.save(self.save_path + '/cost_violation_follower.npy', np.array(self.cost_violation_follower, dtype=np.float32))
                if self.lambda_log:
                    np.save(self.save_path + '/lambda_log.npy', np.array(self.lambda_log, dtype=np.float32))
                if self.cost_violation_log:
                    np.save(self.save_path + '/cost_violation_log.npy', np.array(self.cost_violation_log, dtype=np.float32))
                self.evaluate()
            if self.model_save_rate and (step+1) % self.model_save_rate == 0:
                self.save_models(step=step+1)
        np.save(self.save_path + '/reward_record.npy', self.reward_record)
        if self.lambda_log:
            np.save(self.save_path + '/lambda_log.npy', np.array(self.lambda_log, dtype=np.float32))
        if self.cost_violation_log:
            np.save(self.save_path + '/cost_violation_log.npy', np.array(self.cost_violation_log, dtype=np.float32))
        # final save
        self.save_models(step='final')

    def evaluate(self):
        returns = []
        eval_episodes = getattr(self.args, 'evaluate_episodes', 5)
        ep_len = getattr(self.args, 'evaluate_episode_len', 200)
        for _ in range(eval_episodes):
            s, info = self.eval_env.reset()
            s = _obs_to_agents(s)
            ep_reward = [0.0 for _ in range(self.args.n_agents)]
            for t in range(ep_len):
                obs_cat = np.concatenate([s[0], s[1]], axis=0)
                q1_np, q2_np, g1_np, g2_np, safe_sets = self._build_tables_and_safe_sets(obs_cat)
                a0, a1 = self._mip_action_full(q1_np, q2_np, g1_np, g2_np, safe_sets)
                s_next, r, done, trunc, info = self.eval_env.step((a0, a1))
                s_next = _obs_to_agents(s_next)
                for i in range(self.args.n_agents):
                    ep_reward[i] += r[i]
                s = s_next
                if np.all(done):
                    break
            returns.append(ep_reward)
        mean_ret = np.mean(np.array(returns), axis=0)
        return mean_ret
