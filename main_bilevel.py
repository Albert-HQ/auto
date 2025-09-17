from runner_bilevel import Runner_Bilevel,Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_highway_env
import numpy as np
import json
import os
import sys as _sys


if __name__ == '__main__':
    # get the params
    args = get_args()

    # set train params (可被 config.json 覆盖)
    # 默认示例：用户可通过 --scenario alias 与 config 文件切换
    # 如果开启冒烟测试，后续会强制缩小规模

    # 典型：按场景自动选择默认保存目录；若命令行显式传入 --file-path 则不覆盖
    _cli_has_file_path = any(s.startswith('--file-path') for s in _sys.argv)
    if not _cli_has_file_path:
        scen_base = (args.scenario_name.split('-')[0] if getattr(args, 'scenario_name', None) else 'roundabout')
        args.file_path = f"./{scen_base}_env_result/exp2"
    # 若命令行显式传入 --file-path，则保留 args.file_path 原值

    if not os.path.isabs(args.file_path):
        args.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.file_path)

    # 读取 config.json（存在则覆盖参数）
    config_json = os.path.join(args.file_path, 'config.json')
    if os.path.exists(config_json):
        with open(config_json, 'r',encoding='utf-8') as f:
            vars(args).update(json.load(f))

    # 冒烟测试：缩短时间与单 seed，避免大量运行
    if getattr(args, 'smoke_test', False):
        args.time_steps = min(1000, getattr(args, 'time_steps', 1000))
        args.evaluate_episodes = 2
        seeds = [0]
        # 减小 buffer/batch 避免内存占用
        args.buffer_size = int(5e4)
        args.batch_size = min(64, getattr(args, 'batch_size', 64))
        args.sample_size = min(128, getattr(args, 'sample_size', 128))
        # 连续场景：缩短 episode
        args.max_episode_len = min(100, getattr(args, 'max_episode_len', 100))
    else:
        seeds = [0] ##改动前：[0,1,2] ##

    for i in seeds:
        args.seed = i
        args.save_dir = os.path.join(args.file_path, f"seed_{args.seed}")
        os.makedirs(args.save_dir, exist_ok=True)

        # ---- 约束阈值健壮化与回填 ----
        def _to_float(x, default=0.0):
            if x is None:
                return float(default)
            if isinstance(x, (list, tuple)):
                x = x[0] if len(x) > 0 else default
            try:
                return float(x)
            except Exception:
                return float(default)
        common_ct = getattr(args, 'cost_threshold', None)
        leader_ct = getattr(args, 'cost_threshold_leader', None)
        follower_ct = getattr(args, 'cost_threshold_follower', None)
        common_ct_f = _to_float(common_ct, default=0.0)
        leader_ct_f = _to_float(leader_ct, default=common_ct_f)
        follower_ct_f = _to_float(follower_ct, default=common_ct_f)
        setattr(args, 'cost_threshold', common_ct_f)
        setattr(args, 'cost_threshold_leader', leader_ct_f)
        setattr(args, 'cost_threshold_follower', follower_ct_f)

        # set env
        env, eval_env, args = make_highway_env(args)
        np.random.seed(args.seed)

        # choose action type and algorithm
        if args.action_type == "continuous":
            if args.version == "bilevel":
                runner = Runner_Bilevel(args, env, eval_env)
            elif args.version == "c_bilevel":
                runner = Runner_C_Bilevel(args, env, eval_env)
        elif args.action_type == "discrete":
            runner = Runner_Stochastic(args, env, eval_env)
        else:
            raise ValueError(f"Unknown action_type {args.action_type}")

        # === 新增：显式评估前加载最新（或最终）权重 ===
        if getattr(args, 'evaluate', False):
            if hasattr(runner, 'load_models'):
                try:
                    runner.load_models()  # 连续：占位；离散：载入 latest
                except Exception as e:
                    print('[WARN] load_models error:', e)
            # 额外提示：检查常见权重文件是否存在
            expected = []
            if args.action_type == 'discrete':
                expected = [os.path.join(args.save_dir, 'discrete_model', f) for f in ['q1_latest.pth','q2_latest.pth']]
            else:  # 连续：检查 agent_0/1 目录
                expected = [os.path.join(args.save_dir, 'agent_0', 'actor_params.pkl'), os.path.join(args.save_dir, 'agent_1', 'actor_params.pkl')]
            missing = [p for p in expected if not os.path.exists(p)]
            if missing:
                print('[INFO] Some expected weight files not found, evaluating untrained or partially trained models:', missing)

        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
        else:
            runner.run()

        record_train = getattr(args, 'record_train_video', (getattr(args, 'record_video', False) and not getattr(args, 'evaluate', False)))
        record_eval = getattr(args, 'record_eval_video', (getattr(args, 'record_video', False) and getattr(args, 'evaluate', False)))
        if record_train and hasattr(runner, 'record_video_train'):
            runner.record_video_train()
        if record_eval:
            if hasattr(runner, 'record_video_eval'):
                runner.record_video_eval()
            elif hasattr(runner, 'record_video'):
                runner.record_video()



