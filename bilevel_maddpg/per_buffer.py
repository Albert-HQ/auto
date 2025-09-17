import threading
import numpy as np


class PERBuffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        self.n_agents = args.n_agents
        # 可选：多成本维度（若未提供则为1）
        self.num_costs = getattr(args, 'num_costs', 1)
        # memory management
        self.current_size = 0
        # create the buffer to store info (transition-level)
        self.buffer = {}
        # 观测
        for i in range(self.n_agents):
            self.buffer[f'o_{i}'] = np.empty((self.size, self.args.obs_shape[i]), dtype=np.float32)
            # 动作：离散用 int，连续用向量
            if hasattr(self.args, 'action_dim') and self.args.action_type == 'discrete':
                self.buffer[f'u_{i}'] = np.empty((self.size,), dtype=np.int32)
            else:
                self.buffer[f'u_{i}'] = np.empty((self.size, self.args.action_shape[i]), dtype=np.float32)
            self.buffer[f'r_{i}'] = np.empty((self.size,), dtype=np.float32)
            # 成本：若多成本则存向量
            if self.num_costs == 1:
                self.buffer[f'c_{i}'] = np.empty((self.size,), dtype=np.float32)
            else:
                self.buffer[f'c_{i}'] = np.empty((self.size, self.num_costs), dtype=np.float32)
            self.buffer[f'o_next_{i}'] = np.empty((self.size, self.args.obs_shape[i]), dtype=np.float32)
        self.buffer['done'] = np.empty((self.size,), dtype=np.float32)

        # PER 结构
        self.priorities = np.zeros((self.size,), dtype=np.float32)
        self.p_max = 1.0
        self.lock = threading.Lock()

        # PER 超参
        self.alpha = getattr(args, 'per_alpha', 0.6)
        self.beta = getattr(args, 'per_beta_start', 0.4)
        self.beta_increment = getattr(args, 'per_beta_increment', 0.0)
        self.eps_p = getattr(args, 'per_eps', 1e-6)
        # 统一的优先级裁剪上下限（避免极端爆炸或全部趋同）
        self.p_min_clip = 1e-6
        self.p_max_clip = 1e3

    # 辅助：展平并截断/零填充到目标长度
    def _fit_obs(self, arr, target_len):
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size == target_len:
            return x
        if x.size > target_len:
            return x[:target_len]
        out = np.zeros((target_len,), dtype=np.float32)
        out[:x.size] = x
        return out

    # store a single transition (transition-level storage)
    def store_episode(self, o, u, r, o_next, u_next, t, c=0):
        """存一条 transition。
        o, o_next: List[obs_i]
        u: List[action_i]; r: List[float]; c: List[cost_i] or scalar; t: bool/int done
        """
        idx = self._get_storage_idx(inc=1)
        for i in range(self.n_agents):
            # 观测写入（展平并匹配长度）
            self.buffer[f'o_{i}'][idx] = self._fit_obs(o[i], self.args.obs_shape[i])
            self.buffer[f'o_next_{i}'][idx] = self._fit_obs(o_next[i], self.args.obs_shape[i])
            # 奖励写入
            ri = r[i] if isinstance(r, (list, tuple, np.ndarray)) else r
            self.buffer[f'r_{i}'][idx] = float(ri)
            # 成本写入
            if self.num_costs == 1:
                if isinstance(c, (list, tuple, np.ndarray)) and len(c) == self.n_agents:
                    ci = c[i]
                else:
                    ci = c
                self.buffer[f'c_{i}'][idx] = float(np.asarray(ci, dtype=np.float32).sum())
            else:
                self.buffer[f'c_{i}'][idx] = np.asarray(c[i], dtype=np.float32).reshape(-1)[:self.num_costs]
            # 动作
            if hasattr(self.args, 'action_dim') and self.args.action_type == 'discrete':
                self.buffer[f'u_{i}'][idx] = int(u[i])
            else:
                self.buffer[f'u_{i}'][idx] = np.asarray(u[i], dtype=np.float32).reshape(-1)[:self.args.action_shape[i]]
        self.buffer['done'][idx] = float(t)
        # 新样本赋最大优先级，确保被采样（裁剪）
        self.priorities[idx] = np.clip(self.p_max, self.p_min_clip, self.p_max_clip)

    def sample(self, batch_size, stratified=True):
        assert self.current_size > 0, 'Buffer is empty'
        size = self.current_size
        scaled_p = np.clip(self.priorities[:size], self.p_min_clip, self.p_max_clip) ** self.alpha
        if scaled_p.sum() == 0:
            scaled_p[:] = 1.0
        prob = scaled_p / scaled_p.sum()

        if stratified and batch_size > 1:
            segments = np.linspace(0.0, 1.0, batch_size + 1)
            samples = []
            cdf = prob.cumsum()
            for i in range(batch_size):
                a, b = segments[i], segments[i+1]
                lo = np.searchsorted(cdf, a)
                hi = np.searchsorted(cdf, b)
                if hi <= lo:
                    idx = lo
                else:
                    idx = np.random.randint(lo, hi)
                samples.append(idx)
            idxes = np.array(samples)
        else:
            idxes = np.random.choice(size, batch_size, p=prob)

        with np.errstate(divide='ignore'):
            weights = (size * prob[idxes]) ** (-self.beta)
        weights /= (weights.max() + 1e-8)
        self.beta = min(1.0, self.beta + self.beta_increment)

        temp_buffer = {'idx': idxes, 'weights': weights.astype(np.float32)}
        for k, v in self.buffer.items():
            temp_buffer[k] = v[idxes]
        return temp_buffer

    def update_priorities(self, idxes, p_new):
        """p_new 可以是 TD 组合残差，也可以是已经 (..)^alpha 处理后的 priority。
        统一在此做裁剪，保证数值稳定。
        """
        p_new = np.asarray(p_new).reshape(-1)
        if not getattr(self.args, 'priorities_ready', False):
            p_new = (p_new + self.eps_p) ** self.alpha
        p_new = np.clip(p_new, self.p_min_clip, self.p_max_clip)
        with self.lock:
            self.priorities[idxes] = p_new
            # 更新 p_max 但保持裁剪
            self.p_max = float(np.clip(max(self.p_max, p_new.max()), self.p_min_clip, self.p_max_clip))

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        with self.lock:
            if self.current_size + inc <= self.size:
                idx = np.arange(self.current_size, self.current_size + inc)
            elif self.current_size < self.size:
                overflow = inc - (self.size - self.current_size)
                idx_a = np.arange(self.current_size, self.size)
                idx_b = np.random.randint(0, self.current_size, overflow)
                idx = np.concatenate([idx_a, idx_b])
            else:
                idx = np.random.randint(0, self.size, inc)
            self.current_size = min(self.size, self.current_size + inc)
            if inc == 1:
                idx = idx[0]
        return idx
