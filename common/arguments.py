import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Bilevel Reinforcement Learning for highway environments")
    # Environment
    parser.add_argument("--file-path", type=str, default="./roundabout_env_result/exp1", help="file path for reading config and saving result")
    parser.add_argument("--scenario-name", type=str, default="roundabout-v0", help="name of the scenario script")
    # 新增：友好别名，支持使用 --scenario intersection 形式
    parser.add_argument("--scenario", type=str, default=None, help="alias of scenario-name: intersection|racetrack|merge|roundabout")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000, help="number of time steps")
    parser.add_argument("--action-type", type=str, default="continuous", help="action type")
    parser.add_argument("--version", type=str, default="c_bilevel", help="version of algorithm")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # Core training parameters
    parser.add_argument("--lagrangian_multiplier", type=float, default=1, help="initial value of lagrangian multiplier")
    parser.add_argument("--lagrangian_max_bound", type=float, default=20, help="max bound of lagrangian multiplier")
    parser.add_argument("--cost_threshold", type=float, default=2, help="threshold of cost")
    # Optional separate thresholds for leader/follower (d1, d2). If not set, fallback to cost_threshold.
    parser.add_argument("--cost_threshold_leader", type=float, default=None, help="leader cost threshold d1")
    parser.add_argument("--cost_threshold_follower", type=float, default=None, help="follower cost threshold d2")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-lagrangian", type=float, default=1e-3, help="learning rate of lagrangian multiplier")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="min epsilon greedy")
    parser.add_argument("--min_noise_rate", type=float, default=0.05, help="min noise rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--sample-size", type=int, default=512, help="number of episodes to optimize at the same time")
    parser.add_argument("--update-rate", type=int, default=10, help="target network update rate")
    parser.add_argument("--enable-cost", type=bool, default=False, help="enable cost constraint")

    # PER hyperparameters and training updates per env step (K)
    parser.add_argument("--per_alpha", type=float, default=0.6, help="PER alpha")
    parser.add_argument("--per_beta_start", type=float, default=0.4, help="PER beta start")
    parser.add_argument("--per_beta_increment", type=float, default=0.0, help="PER beta increment per sample call")
    parser.add_argument("--per_eps", type=float, default=1e-6, help="PER small epsilon for priorities")
    parser.add_argument("--updates-per-step", type=int, default=1, help="K critic updates per env step")
    parser.add_argument("--use-mip", action="store_true", help="use MIP action selection for discrete runner")
    # PER priority mixing weights (lambda)
    parser.add_argument("--per_lambda_q1", type=float, default=1.0, help="lambda for |dQ1|")
    parser.add_argument("--per_lambda_q2", type=float, default=1.0, help="lambda for |dQ2|")
    parser.add_argument("--per_lambda_g1", type=float, default=1.0, help="lambda for |dG1|")
    parser.add_argument("--per_lambda_g2", type=float, default=1.0, help="lambda for |dG2|")

    # MIP solver hyperparameters
    parser.add_argument("--mip_big_m", type=float, default=1e3, help="big-M used in MIP constraints")
    parser.add_argument("--mip_timeout", type=float, default=None, help="time limit (seconds) for MIP solver, None for unlimited")
    # 新增：MIP 求解器选择（gurobi 或 pulp）及其参数
    parser.add_argument("--mip_solver", type=str, default="gurobi", help="MIP solver to use: 'gurobi' or 'pulp'")
    parser.add_argument("--gurobi_threads", type=int, default=None, help="Number of threads for Gurobi (None = default)")

    # CS-MATD3 (continuous) hyperparameters
    parser.add_argument("--use-cs-matd3", action="store_true", help="enable CS-MATD3 updates in Runner_C_Bilevel")
    parser.add_argument("--policy-delay", type=int, default=2, help="policy update delay for CS-MATD3")
    parser.add_argument("--target-noise-sigma", type=float, default=0.2, help="target policy smoothing noise std")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="target policy smoothing noise clip")
    parser.add_argument("--expl-sigma", type=float, default=0.1, help="exploration noise std for action selection")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./roundabout_env_result/exp1", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=300, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--record-video", type=bool, default=False, help="record video")
    # 新增：冒烟测试开关
    parser.add_argument("--smoke-test", action="store_true", help="enable smoke test: shrink time steps & seeds")
    args = parser.parse_args()

    # 场景别名映射
    if args.scenario is not None:
        mapping = {
            'intersection': 'intersection-v0',
            'racetrack': 'racetrack-v0',
            'merge': 'merge-v0',
            'roundabout': 'roundabout-v0'
        }
        if args.scenario in mapping:
            args.scenario_name = mapping[args.scenario]

    return args
