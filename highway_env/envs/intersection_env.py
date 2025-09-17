from typing import Dict, Tuple, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle


class IntersectionEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": [4.5, 9]
            },
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False,
            # 新增：受控车辆到达/离开后循环重生（仅用于可视化）
            "recycle_on_arrival": True,
            # 新增：低速超时回收（防长时间僵滞）
            "recycle_if_stuck": True,
            "stuck_speed_threshold": 0.3,      # m/s
            "stuck_time_seconds": 3.0,         # s
            # 新增：固定将受控车的目的地设置为对向出口（直行），与“仅纵向控制”相匹配
            "fixed_straight_destinations": True
        })
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
                   ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived(vehicle) or
                self.time >= self.config["duration"])

    def _is_truncated(self) -> bool:
        return

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        # info["leader_obs"] = [self.controlled_vehicles[0].lane.local_coordinates(self.controlled_vehicles[0].position)[0], self.controlled_vehicles[0].speed]
        # info["follower_obs"] = [self.controlled_vehicles[1].lane.local_coordinates(self.controlled_vehicles[1].position)[0], self.controlled_vehicles[1].speed]
        
        return info

    def _reset(self) -> None:
        self.leader_arrived = False
        self.follower_arrived = False
        self.first_arrived = 0
        # 新增：缓存上一步是否已终止，用于在终止后的步数将奖励置0，避免重复累计
        self._leader_done_prev = False
        self._follower_done_prev = False
        # 新增：低速计时器（在创建车辆后初始化长度）
        self._stuck_counters = []
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        # 初始化与受控车辆数一致的计时器
        self._stuck_counters = [0.0 for _ in self.controlled_vehicles]

    # def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     self._clear_vehicles()
    #     self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
    #     return obs, reward, terminated, truncated, info

    def leader_agend_reward(self, vehicle):
        """Per-agent per-objective reward signal."""
        # scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # rewards =  {
        #     "collision_reward": vehicle.crashed,
        #     "high_speed_reward": np.clip(scaled_speed, 0, 1),
        #     "arrived_reward": self.has_arrived_target(vehicle, 2),
        #     "on_road_reward": vehicle.on_road
        # }

        reward = 0

        if vehicle.crashed:
            reward -= 5
        
        if vehicle.speed >=8 and vehicle.speed <=12:
            reward += 2
    
        # if vehicle.on_road:
        #     reward += 1
        
        if self.has_arrived_target(vehicle, 2):
            if self.leader_arrived == False and self.follower_arrived == False:
                reward += 5
                self.first_arrived = 1
            elif self.leader_arrived == False and self.follower_arrived == True:
                reward += 2
            self.leader_arrived = True
        return reward
    
    def follower_agend_reward(self, vehicle):
        """Per-agent per-objective reward signal."""
        # scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # rewards =  {
        #     "collision_reward": vehicle.crashed,
        #     "high_speed_reward": np.clip(scaled_speed, 0, 1),
        #     "arrived_reward": self.has_arrived_target(vehicle, 3),
        #     "on_road_reward": vehicle.on_road
        # }
    
        reward = 0

        if vehicle.crashed:
            reward -= 5
        
        if vehicle.speed >=8 and vehicle.speed <=12:
            reward += 2
    
        # if vehicle.on_road:
        #     reward += 1
        
        if self.has_arrived_target(vehicle, 3):
            if self.leader_arrived == False and self.follower_arrived == False:
                reward += 5
                self.first_arrived = 2
            elif self.leader_arrived == True and self.follower_arrived == False:
                reward += 2
            self.follower_arrived = True
        
        return reward
    
    def leader_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived_target(vehicle, 2) or
                self.time >= self.config["duration"] or
                (self.config["offroad_terminal"] and not vehicle.on_road))
    
    def follower_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived_target(vehicle, 3) or
                self.time >= self.config["duration"] or
                (self.config["offroad_terminal"] and not vehicle.on_road))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # simulation
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        
        # observation
        obs = self.observation_type.observe()
        
        truncated = self._is_truncated()
        self._clear_vehicles()

        # terminate（当前步是否终止）
        leader_terminated_now = self.leader_is_terminal(self.controlled_vehicles[0])
        follower_terminated_now = self.follower_is_terminal(self.controlled_vehicles[1])
        # 锁存终止：一旦终止则保持 True，直到 reset
        leader_terminated = leader_terminated_now or self._leader_done_prev
        follower_terminated = follower_terminated_now or self._follower_done_prev
        terminated = [leader_terminated, follower_terminated]

        # reward
        reward = np.zeros(2)
        leader_reward = self.leader_agend_reward(self.controlled_vehicles[0])
        follower_reward = self.follower_agend_reward(self.controlled_vehicles[1])   
        # 若上一步已终止，则本步不再累计奖励（避免碰撞罚一直累加）
        if self._leader_done_prev:
            leader_reward = 0
        if self._follower_done_prev:
            follower_reward = 0
        reward = [leader_reward, follower_reward]       

        # info
        info = self._info(obs, action)
        info["first_arrived"] = self.first_arrived
        info["crashed"] = self.controlled_vehicles[0].crashed or self.controlled_vehicles[1].crashed
        # info["leader_arrived"] = self.has_arrived_target(self.controlled_vehicles[0], 2)
        # info["follower_arrived"] = self.has_arrived_target(self.controlled_vehicles[1], 3)

        # cost
        cost = np.zeros(2)
        cost[0] += 5*self.controlled_vehicles[0].crashed
        cost[1] += 5*self.controlled_vehicles[1].crashed
        info["cost"] = cost

        # 更新上一时刻的done缓存（锁存）
        self._leader_done_prev = leader_terminated
        self._follower_done_prev = follower_terminated

        # 可视化用：循环重生受控车辆，避免驶离视野
        if self.config.get("recycle_on_arrival", True):
            self._recycle_controlled_vehicles()
        # 若车辆长时间低速（未终止）则回收，避免僵滞
        if self.config.get("recycle_if_stuck", True):
            self._unstick_controlled_vehicles()

        return obs, reward, terminated, truncated, info
    

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Controlled vehicles
        self.controlled_vehicles = []
        fixed_straight = bool(self.config.get("fixed_straight_destinations", False))
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o" + str(ego_id % 4), "ir" + str(ego_id % 4), 0))
            if fixed_straight:
                # 对向出口：直行，不需要转向；与仅纵向控制配置兼容
                dest_idx = (ego_id + 2) % 4
                destination = f"o{dest_idx}"
            else:
                destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 1 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60)
            )
            try:
                ego_vehicle.plan_route_to(destination)
                # 若是离散元动作车辆，初始化巡航速度索引
                if hasattr(ego_vehicle, 'speed_to_index'):
                    ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                    ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
                else:
                    ego_vehicle.target_speed = ego_lane.speed_limit
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=8 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 5) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
    
    def has_arrived_target(self, vehicle: Vehicle, road_i: int, exit_distance: float = 5) -> bool:
        #print(vehicle.lane.local_coordinates(vehicle.position)[0])
        return "il"+str(road_i) in vehicle.lane_index[0] \
               and "o"+str(road_i) in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    # 新增：将已到达/离开/撞击的受控车辆循环重生回入口，避免一直驶离屏幕
    def _recycle_controlled_vehicles(self) -> None:
        try:
            for i, veh in enumerate(self.controlled_vehicles):
                # 判断是否离开出口
                def _is_leaving(v: Vehicle) -> bool:
                    try:
                        return ("il" in v.lane_index[0] and "o" in v.lane_index[1] and
                                v.lane.local_coordinates(v.position)[0] >= v.lane.length - 4 * v.LENGTH)
                    except Exception:
                        return False
                # 仅当该智能体已终止（锁存），或确实离开/越界时才回收
                agent_done = self._leader_done_prev if i == 0 else (self._follower_done_prev if i == 1 else False)
                need_recycle = agent_done or (not getattr(veh, 'on_road', True)) or _is_leaving(veh)
                if not need_recycle:
                    continue
                # 复位到对应入口车道并设定直行至对向出口
                ego_lane = self.road.network.get_lane((f"o{i % 4}", f"ir{i % 4}", 0))
                dest_idx = (i + 2) % 4
                s0 = 60 + 1 * self.np_random.normal(1)
                pos = ego_lane.position(s0, 0)
                heading = ego_lane.heading_at(s0)
                speed = ego_lane.speed_limit
                veh.position = pos
                veh.heading = heading
                try:
                    veh.speed = speed
                except Exception:
                    pass
                veh.crashed = False
                try:
                    veh.plan_route_to(f"o{dest_idx}")
                except Exception:
                    pass
                # 恢复目标速度（无论离散/连续）
                try:
                    if hasattr(veh, 'speed_to_index'):
                        veh.speed_index = veh.speed_to_index(speed)
                        veh.target_speed = veh.index_to_speed(veh.speed_index)
                    else:
                        veh.target_speed = speed
                except Exception:
                    pass
        except Exception:
            # 若任何异常，不影响环境运行
            pass

    # 新增：低速卡滞检测与回收，防止车辆长时间不动
    def _unstick_controlled_vehicles(self) -> None:
        try:
            # 保护：计数器长度对齐
            if len(self._stuck_counters) != len(self.controlled_vehicles):
                self._stuck_counters = [0.0 for _ in self.controlled_vehicles]
            dt = 1.0 / max(1, int(self.config.get("policy_frequency", 1)))
            speed_thr = float(self.config.get("stuck_speed_threshold", 0.3))
            timeout = float(self.config.get("stuck_time_seconds", 3.0))
            for i, veh in enumerate(self.controlled_vehicles):
                # 已终止的agent或不在路上，计数清零并跳过（由 recycle_on_arrival 处理）
                agent_done = self._leader_done_prev if i == 0 else (self._follower_done_prev if i == 1 else False)
                if agent_done or not getattr(veh, 'on_road', True):
                    self._stuck_counters[i] = 0.0
                    continue
                v = float(getattr(veh, 'speed', 0.0) or 0.0)
                if v < speed_thr:
                    self._stuck_counters[i] += dt
                else:
                    self._stuck_counters[i] = 0.0
                if self._stuck_counters[i] >= timeout:
                    # 回收该车辆到入口
                    ego_lane = self.road.network.get_lane((f"o{i % 4}", f"ir{i % 4}", 0))
                    dest_idx = (i + 2) % 4
                    s0 = 60 + 1 * self.np_random.normal(1)
                    pos = ego_lane.position(s0, 0)
                    heading = ego_lane.heading_at(s0)
                    speed_lim = ego_lane.speed_limit
                    veh.position = pos
                    veh.heading = heading
                    try:
                        veh.speed = speed_lim
                    except Exception:
                        pass
                    veh.crashed = False
                    try:
                        veh.plan_route_to(f"o{dest_idx}")
                    except Exception:
                        pass
                    try:
                        if hasattr(veh, 'speed_to_index'):
                            veh.speed_index = veh.speed_to_index(speed_lim)
                            veh.target_speed = veh.index_to_speed(veh.speed_index)
                        else:
                            veh.target_speed = speed_lim
                    except Exception:
                        pass
                    # 清零计数器
                    self._stuck_counters[i] = 0.0
        except Exception:
            # 安全兜底，不影响主流程
            pass

class ContinuousIntersectionEnv(IntersectionEnv):
    """连续动作版本的路口环境，对应注册 id: intersection-v1
    仅改写默认配置中的 action 类型为 ContinuousAction，保持其余逻辑复用 IntersectionEnv。
    用途：CS-MATD3 / 约束 Stackelberg 在路口连续控制场景的冒烟测试。
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        # 覆盖为连续动作（仅纵向，避免横向控制导致路线偏离）
        config["action"] = {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": False
        }
        # 连续控制通常需要更高的 policy_frequency（沿用默认即可，如需可在 env_config.json 覆盖）
        return config
