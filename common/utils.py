import numpy as np
import gym
import json
import os
import highway_env
highway_env.register_highway_envs()

# 以 UTF-8 优先读取 JSON，如遇到 BOM 则回退到 UTF-8-SIG
def _load_json_utf8(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)


def make_highway_env(args):
    env = gym.make(args.scenario_name, render_mode='rgb_array')
    eval_env = gym.make(args.scenario_name, render_mode='rgb_array')

    config_path = os.path.join(args.file_path, 'env_config.json')
    cfg = _load_json_utf8(config_path)
    env.configure(cfg)
    eval_env.configure(cfg)
    
    # 兼容调用；gym 0.26 reset 也可返回 obs 或 (obs, info)，此处不使用返回值
    env.reset()
    eval_env.reset()

    args.n_players = 2  # agent number
    args.n_agents = 2  # agent number
    args.obs_shape = [8, 8]  # obs dim
    args.action_shape = [1,1] # act dim
    # highway env
    # args.obs_shape = 20 
    # args.action_shape = 1
    args.action_dim = [5,5] # act num for discrete action
    args.terminal_shape = [1,1] # terminal dim
    args.high_action = 1  # act high for continuous action
    args.low_action = -1  # act low for continuous action

    return env, eval_env, args

