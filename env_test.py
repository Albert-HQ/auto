import gym
import time
import numpy as np
import highway_env
highway_env.register_highway_envs()

env = gym.make("u-turn-v0")
env.configure({ 
    "manual_control": True,
    "real_time_rendering": True,
    "screen_width": 1000,
    "screen_height": 1000,
    "duration": 20,
    "controlled_vehicles": 2,  # 确保有两辆受控车辆
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "flatten": True,
            "absolute": True,
            "see_behind": True,
            "normalize": False,
            "features": ["x", "y", "vx", "vy"],
            "vehicles_count": 2
        }
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction"
        }
    }
})

env.reset()
done = False

while not done:
    act = env.action_space.sample()
    obs, reward, done_flags, _, _ = env.step(act)  # Gym 多车辆返回 done 列表
    done = np.all(done_flags)  # 所有车辆完成才停止

    # 循环打印每辆受控车辆的目标速度
    for i, vehicle in enumerate(env.controlled_vehicles):
        print(f"Vehicle {i} target speeds:", vehicle.target_speeds)
    print(".......")

    env.render()
    time.sleep(0.1)  # 控制帧率，动画更流畅

env.close()
