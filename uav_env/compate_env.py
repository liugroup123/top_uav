import numpy as np
import os
import sys
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World, Obstacle
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env_2 import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe.interaction_obs import InteractionAttention
from pettingzoo.mpe.hierarchical_obs import RobotEnvironment, MultiGATNet
from pettingzoo.mpe.gst_lstm import GST_LSTM  

import torch
from torch_geometric.data import Batch
import networkx as nx
from sklearn.cluster import KMeans


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N_drones=5,
            N_targets=10,
            N_obstacles=4,  # 障碍物数量
            coverage_radius=0.5,
            local_ratio=0.5,  # 可以理解为这个参数是平衡全局奖励和单个智能体的奖励
            max_cycles=300,  # 这个设置大一点，一般来说要大于外部的时间步
            continuous_actions=True,  # 定义连续或者离散动作空间
            render_mode=None,
    ):
        EzPickle.__init__(
            self,
            N_drones=N_drones,
            N_targets=N_targets,
            N_obstacles=N_obstacles,
            coverage_radius=coverage_radius,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        scenario = Scenario(
            N_drones=N_drones, N_targets=N_targets, N_obstacles=N_obstacles, coverage_radius=coverage_radius,
        )
        world = scenario.make_world()

        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "drone_coverage_env"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self, N_drones, N_targets, N_obstacles, coverage_radius):
        super().__init__()
        self.N_drones = N_drones
        self.N_targets = N_targets
        self.N_obstacles = N_obstacles
        self.coverage_radius = coverage_radius
        self.communication_radius = 1.0
        self.is_drone_env = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_model = None
        self.gat_model = None
        self.attention_model = None

        self.initial_threshold = 0.75
        self.max_threshold = 0.95
        self.threshold_increase_rate = 0.0005
        self.training_step = 0
        self.stability_bonus_value = 800

        # 用于记录已覆盖的目标点
        self.covered_targets = set()

    def make_world(self):
        world = World()
        world.dim_c = 2
        world.collaborative = True

        # 添加无人机 (agents)
        world.agents = [Agent() for _ in range(self.N_drones)]
        for i, agent in enumerate(world.agents):
            agent.name = f"drone_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.03
            agent.accel = 2.0
            agent.max_speed = 1.0
            agent.color = np.array([1.0, 1.0, 0.6])
            agent.communication_radius = self.communication_radius
            agent.detection_radius = 0.5
            agent.target = [0.0, 0.0]
            agent.no_fly = False  # 禁飞区属性

        # 添加地标 (landmarks) 和障碍物 (obstacles)
        world.landmarks = [Landmark() for _ in range(self.N_targets)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"target_{i}"
            landmark.is_obstacle = False
            landmark.size = 0.015
            landmark.color = np.array([0.5, 1.0, 0.5])  # 浅绿色

            # 通用属性
            landmark.collide = True
            landmark.movable = False
            landmark.boundary = False

        world.obstacles = [Obstacle() for _ in range(self.N_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle_{i}"
            obstacle.size = np.random.uniform(0.10, 0.15)  # 随机障碍物大小
            obstacle.color = np.array([0.25, 0.25, 0.25])  # 灰色

            # 相关通用属性
            obstacle.collide = False
            obstacle.movable = True
            obstacle.boundary = False
            obstacle.no_fly = True

        self.env_model = RobotEnvironment(world)
        self.gat_model = MultiGATNet(
            neighbor_input_dim=4,
            target_input_dim=2,
            obstacle_input_dim=3,
            obstacle_hidden_dim=32,
            hidden_dim=32,
            obstacle_output_dim=20,
            output_dim=20,
            heads=1
        ).to(self.device)

        self.attention_model = InteractionAttention(
            input_dim=60,
            hidden_dim=64,
            output_dim=60,
            num_heads=1
        ).to(self.device)

        # 初始化GST-LSTM模块
        self.gst_lstm_model = GST_LSTM(
            input_dim=60,
            hidden_dim=64,
        ).to(self.device)

        # 设置评估模式
        self.gat_model.eval()
        self.attention_model.eval()
        self.gst_lstm_model.eval()

        return world

        # 这个是每一轮重置
    def reset_world(self, world, np_random):
        # 设置最底部的纵坐标值
        bottom_y_coord = -0.85  # 假设地图的最底部在 y = -1 处

        # 初始化 Agent 的随机位置，并确保 Agent 之间不重叠且不进入禁飞区
        for i, agent in enumerate(world.agents):
            while True:  # 循环生成位置，直到满足条件
                # 横坐标随机，纵坐标固定为 bottom_y_coord
                potential_pos = np.array([np_random.uniform(-1, 1), bottom_y_coord])
                overlap = False

                # 检查与其他 Agents 是否重叠
                for other_agent in world.agents[:i]:  # 只检查已生成的 Agents
                    distance = np.linalg.norm(potential_pos - other_agent.state.p_pos)
                    if distance < (agent.size + other_agent.size):  # 避免重叠
                        overlap = True
                        break

                # 检查与障碍物是否重叠（禁飞区检查）
                for obstacle in world.obstacles:
                    # 检查 obstacle.state.p_pos 是否为 None，如果是，初始化它
                    if obstacle.state.p_pos is None:
                        obstacle.state.p_pos = np.zeros(world.dim_p)  # 初始化位置为 (0,0)

                    distance = np.linalg.norm(potential_pos - obstacle.state.p_pos)
                    if distance < (agent.size + obstacle.size):  # 避免进入禁飞区
                        overlap = True
                        break

                if not overlap:
                    agent.state.p_pos = potential_pos
                    break

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 初始化目标点 (landmarks) 的随机位置，并确保与 Agent 和其他目标点不重叠
        for i, landmark in enumerate(world.landmarks):
            while True:  # 循环生成位置，直到满足条件
                potential_pos = np_random.uniform(-0.8, +0.8, world.dim_p)
                overlap = False

                # 检查与其他目标点是否重叠
                for other_landmark in world.landmarks[:i]:  # 只检查已生成的目标点
                    distance = np.linalg.norm(potential_pos - other_landmark.state.p_pos)
                    if distance < (landmark.size + other_landmark.size):  # 避免目标点之间重叠
                        overlap = True
                        break

                if not overlap:
                    landmark.state.p_pos = potential_pos
                    break

            landmark.state.p_vel = np.zeros(world.dim_p)

        # 初始化障碍物 (obstacles) 的随机位置，并确保与 Agent 和目标点不重叠
        for i, obstacle in enumerate(world.obstacles):
            while True:  # 循环生成位置，直到满足条件
                potential_pos = np_random.uniform(-0.75, +0.75, world.dim_p)
                overlap = False

                # 检查与所有 Agents 是否重叠
                for agent in world.agents:
                    distance = np.linalg.norm(potential_pos - agent.state.p_pos)
                    if distance < (obstacle.size + agent.size):  # 避免与 Agent 重叠
                        overlap = True
                        break

                # 检查与其他障碍物是否重叠
                for other_obstacle in world.obstacles[:i]:  # 只检查已生成的障碍物
                    distance = np.linalg.norm(potential_pos - other_obstacle.state.p_pos)
                    if distance < (obstacle.size + other_obstacle.size):  # 避免障碍物之间重叠
                        overlap = True
                        break

                if not overlap:
                    obstacle.state.p_pos = potential_pos
                    break

            obstacle.velocity = np_random.uniform(-0.1, 0.1, world.dim_p)
            obstacle.state.p_vel = obstacle.velocity.copy()

    def calculate_unique_coverage_rate(self, world):
        """
        计算唯一覆盖率，即至少有一个无人机覆盖的目标点占总目标点的比例。
        """
        covered_targets = set()
        for agent in world.agents:
            for landmark in world.landmarks:
                if not landmark.is_obstacle:
                    if np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) < self.coverage_radius:
                        covered_targets.add(landmark.name)
        return len(covered_targets) / len(world.landmarks) if len(world.landmarks) > 0 else 0

    def compute_rewards_and_distances(self, world):
        """
        计算无人机与障碍物的边缘距离以及奖励。
        参数:
            world (World): 当前的世界对象，包含无人机和障碍物的位置与半径信息。
        返回:
            reward (float): 总奖励
            distances (list): 每个无人机与障碍物的边缘距离
        """
        coll_reward = 0.0
        distances = []

        collision_threshold = 0.01    # 边缘距离开始有惩罚
        proximity_penalty_factor = 15  # 惩罚系数
        collision_penalty = 15        # 碰撞惩罚

        for drone in world.agents:
            drone_position = drone.state.p_pos  # 获取无人机位置
            drone_size = drone.size  # 获取无人机半径

            for obstacle in world.obstacles:
                obstacle_position = obstacle.state.p_pos  # 获取障碍物位置
                obstacle_size = obstacle.size  # 获取障碍物半径

                # 计算无人机与障碍物的中心距离
                distance_to_center = np.linalg.norm(drone_position - obstacle_position)

                # 计算无人机与障碍物的边缘距离
                edge_distance = distance_to_center - (drone_size + obstacle_size)

                # 保存每个无人机与障碍物的边缘距离
                distances.append({
                    'drone': drone.name,
                    'obstacle': obstacle.name,
                    'edge_distance': edge_distance
                })

                # 接近障碍物的惩罚
                if edge_distance < collision_threshold:
                    proximity_penalty = proximity_penalty_factor * (collision_threshold - edge_distance)
                    coll_reward -= proximity_penalty

                # 碰撞惩罚
                if distance_to_center < (drone_size + obstacle_size):
                    coll_reward -= collision_penalty

        return coll_reward, distances

    def stability_reward(self, world):
        """
        计算稳定覆盖的奖励。
        如果覆盖率超过动态阈值且所有无人机速度低于一定阈值，给予稳定奖励。
        """
        stability_reward = 0
        speed_tolerance = 0.1  # 表示速度容忍度，表示低于最大速度的多少
        coverage_rate = self.calculate_unique_coverage_rate(world)  # 获得覆盖率

        # 动态阈值的计算
        dynamic_threshold = min(self.initial_threshold + self.training_step * self.threshold_increase_rate,
                                self.max_threshold)

        # 用于检查是否所有智能体都符合条件
        all_agents_stable = True

        if coverage_rate >= dynamic_threshold:
            for agent in world.agents:
                # 获取当前智能体的速度和最大速度
                current_speed = np.linalg.norm(agent.state.p_vel)
                max_speed = agent.max_speed

                # 检查该智能体是否满足速度条件
                if current_speed > max_speed * speed_tolerance:
                    all_agents_stable = False  # 若任一智能体不满足条件，则设为 False
                    break  # 退出循环，不再检查其他智能体

            # 如果所有智能体都满足稳定条件，给予整体奖励
            if all_agents_stable:
                stability_reward += self.stability_bonus_value  # 设置一个较大的奖励值

        return stability_reward

    def global_reward(self, world):
        # 增加训练步数计数器
        self.training_step += 1

        # Step 1: 计算代数连通度
        G = nx.Graph()
        for i, agent in enumerate(world.agents):
            G.add_node(i)
        for i, agent in enumerate(world.agents):
            for j, other_agent in enumerate(world.agents):
                if i >= j:
                    continue
                dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                if dist <= getattr(agent, 'communication_radius', 1.0):
                    G.add_edge(i, j)

        try:
            λ2 = nx.algebraic_connectivity(G)
        except:
            λ2 = 0  # 如果图不连通，则连通度为0

        # Step 2: 计算连通性奖励 (r_c)
        if λ2 == 0:
            r_c = -3
        elif λ2 < 0.2:
            r_c = -0.5
        else:
            r_c = 0

        # Step 3: 计算唯一覆盖率和距离奖励 (r_s_d)
        coverage_rate = self.calculate_unique_coverage_rate(world)
        agent_positions = np.array([agent.state.p_pos for agent in world.agents])
        landmark_positions = np.array([landmark.state.p_pos for landmark in world.landmarks])
        distances = np.linalg.norm(agent_positions[:, None, :] - landmark_positions[None, :, :], axis=2)
        min_distances = np.min(distances, axis=0)
        avg_min_distance = np.mean(min_distances)
        clipped_avg_min_distance = np.clip(avg_min_distance, 0, 15)

        r_s_d =  (coverage_rate ** 1.5) * clipped_avg_min_distance

        # Step 4: 计算复合奖励 (r^1)
        k_1 = 35 * len(world.agents)  # 覆盖奖励权重设置为20
        r_1 = r_c + k_1 * r_s_d

        # Step 5: 稳定覆盖奖励 + 碰撞惩罚
        stability_bonus = self.stability_reward(world) * 0.5  # 调整权重
        coll_reward, _ = self.compute_rewards_and_distances(world)

        # 最终全局奖励：距离奖励相关 + 稳定奖励 + 碰撞惩罚奖励相关
        global_reward = r_1 + stability_bonus + coll_reward

        return global_reward / 100

    def reward(self, agent, world):
        rew = 0

        # Step 1: 计算唯一覆盖奖励
        unique_coverage_reward = 0.0
        for landmark in world.landmarks:
            if not landmark.is_obstacle and np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) < self.coverage_radius:
                # 检查是否只有当前无人机覆盖该目标点
                covered_by_others = False
                for other_agent in world.agents:
                    if other_agent is agent:
                        continue
                    if np.linalg.norm(other_agent.state.p_pos - landmark.state.p_pos) < self.coverage_radius:
                        covered_by_others = True
                        break
                if not covered_by_others and landmark.name not in self.covered_targets:
                    unique_coverage_reward += 1.0  # 每个唯一覆盖的目标点奖励1.0
                    self.covered_targets.add(landmark.name)  # 更新覆盖状态
        rew += unique_coverage_reward * 30.0  # 覆盖奖励权重设置为20

        # Step 2: 现有覆盖奖励
        k_reward = 0.5
        max_reward = 10.0
        if hasattr(agent, 'target') and agent.target is not None:
            dist = np.linalg.norm(agent.state.p_pos - agent.target)
            clipped_dist = np.clip(dist, 0.1, 15)
            reward = k_reward * (1 / clipped_dist)
            rew += min(reward, max_reward)
            if dist == 0:
                rew += 2.0

        # Step 3: 边界碰撞惩罚
        rew += self.calculate_boundary_penalty(agent)

        # Step 4: 无人机之间的距离奖励/惩罚
        detection_radius = getattr(agent, 'detection_radius', 0.5)
        min_radius = detection_radius
        max_radius = 2 * detection_radius

        for other_agent in world.agents:
            if other_agent is agent:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
            if dist < 0.75 * min_radius:
                rew -= 2
            elif min_radius <= dist <= max_radius:
                rew += 1.0

        return rew / 100

    def calculate_boundary_penalty(self, agent):
        """
        计算无人机接近或超出边界时的惩罚，使用指数函数形式，并限制惩罚值的上限。
        """
        penalty = 0.0
        max_penalty = 500  # 定义惩罚值的上限，根据需要调整
        boundary_limit = 1.0  # 边界的限制（假设环境在 -1 到 1 的范围内）
        safe_margin = 0.1     # 安全距离，当无人机距离边界小于该值时开始惩罚
        boundary_penalty_factor = 100  # 惩罚因子
        boundary_penalty_exponent = 5  # 指数函数的斜率，用于控制惩罚增长速度

        for dim in range(len(agent.state.p_pos)):
            pos = agent.state.p_pos[dim]

            # 检查负方向边界（接近边界）
            if pos - agent.size < -boundary_limit + safe_margin:
                distance_to_boundary = (-boundary_limit + safe_margin) - (pos - agent.size)
                if distance_to_boundary > 0:
                    # 指数惩罚：随着距离接近0，惩罚迅速增加
                    penalty -= boundary_penalty_factor * np.exp(
                        boundary_penalty_exponent * (safe_margin - distance_to_boundary)
                    )
                    penalty = np.clip(penalty, -max_penalty, 0)

            # 检查负方向边界（超出边界）
            if pos - agent.size < -boundary_limit:
                distance_outside = (-boundary_limit) - (pos - agent.size)
                # 指数惩罚：随着超出距离增加，惩罚迅速增加
                penalty -= boundary_penalty_factor * np.exp(
                    boundary_penalty_exponent * distance_outside
                )
                penalty = np.clip(penalty, -max_penalty*2, 0)

            # 检查正方向边界（接近边界）
            if pos + agent.size > boundary_limit - safe_margin:
                distance_to_boundary = (boundary_limit - safe_margin) - (pos + agent.size)
                if distance_to_boundary < 0:
                    # 指数惩罚：随着距离接近0，惩罚迅速增加
                    penalty -= boundary_penalty_factor * np.exp(
                        boundary_penalty_exponent * (safe_margin - abs(distance_to_boundary))
                    )
                penalty = np.clip(penalty, -max_penalty, 0)

            # 检查正方向边界（超出边界）
            if pos + agent.size > boundary_limit:
                distance_outside = (pos + agent.size) - boundary_limit
                # 指数惩罚：随着超出距离增加，惩罚迅速增加
                penalty -= boundary_penalty_factor * np.exp(
                    boundary_penalty_exponent * distance_outside
                )
                penalty = np.clip(penalty, -max_penalty*2, 0)

        return penalty

    def observation(self, agent, world):
        device = self.device

        max_neighbors = 4
        num_agents = len(world.agents)

        # Step 1: 计算所有机器人的嵌入表示 (GAT 模块)
        robot_embeddings = []
        for other_agent in world.agents:
            neighbor_graph_data = self.env_model.build_robot_neighbor_graph(other_agent)
            target_graph_data = self.env_model.build_robot_target_graph(other_agent)
            obstacle_graph_data = self.env_model.build_obstacle_neighbor_graph(other_agent)

            # 确保数据在正确的设备上，并且类型正确
            neighbor_graph_data.x = neighbor_graph_data.x.to(dtype=torch.float32, device=device, non_blocking=True)
            neighbor_graph_data.edge_index = neighbor_graph_data.edge_index.to(dtype=torch.long, device=device,
                                                                               non_blocking=True)
            target_graph_data.x = target_graph_data.x.to(dtype=torch.float32, device=device, non_blocking=True)
            target_graph_data.edge_index = target_graph_data.edge_index.to(dtype=torch.long, device=device,
                                                                           non_blocking=True)
            obstacle_graph_data.x = obstacle_graph_data.x.to(dtype=torch.float32, device=device, non_blocking=True)
            obstacle_graph_data.edge_index = obstacle_graph_data.edge_index.to(dtype=torch.long, device=device,
                                                                               non_blocking=True)

            # 修改这里：使用 torch.autocast
            with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=device == 'cuda'):
                robot_embedding = self.gat_model(neighbor_graph_data, target_graph_data, obstacle_graph_data)
            robot_embeddings.append(robot_embedding)

        observation_embeddings = torch.stack(robot_embeddings).to(device)  # (num_robots, 60)

        # Step 2: 获取邻居索引并构造邻接矩阵
        neighbors_indices = torch.zeros((num_agents, max_neighbors), dtype=torch.long, device=device)
        adj_matrix = torch.zeros((num_agents, num_agents), dtype=torch.float32, device=device)

        for i, other_agent in enumerate(world.agents):
            neighbor_agents = self.env_model.get_neighbors(other_agent)[:max_neighbors]
            neighbor_indices = [world.agents.index(neigh) for neigh in neighbor_agents]
            padding_size = max_neighbors - len(neighbor_indices)
            if padding_size > 0:
                neighbor_indices += [0] * padding_size
            neighbors_indices[i] = torch.tensor(neighbor_indices, dtype=torch.long, device=device)

            for neigh in neighbor_agents:
                j = world.agents.index(neigh)
                adj_matrix[i, j] = 1.0

        # Step 3: 使用 InteractionAttention 计算交互嵌入
        with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=device == 'cuda'):
            interaction_embeddings = self.attention_model(observation_embeddings, neighbors_indices)

        # Step 4: 通过 GST-LSTM 模块捕获时间依赖
        if not hasattr(self, "Ht_1") or not hasattr(self, "Ct_1") or not hasattr(self, "time_step_counter"):
            hidden_dim = self.gst_lstm_model.hidden_dim
            self.Ht_1 = torch.zeros(num_agents, hidden_dim, device=device)
            self.Ct_1 = torch.zeros(num_agents, hidden_dim, device=device)
            self.time_step_counter = 0

        # 每50个时间步重置隐藏状态和单元记忆
        if self.time_step_counter % 50 == 0:
            self.Ht_1.zero_()
            self.Ct_1.zero_()
        self.time_step_counter += 1

        with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=device == 'cuda'):
            Ht, Ct = self.gst_lstm_model(interaction_embeddings, self.Ht_1, self.Ct_1, adj_matrix.detach())

        self.Ht_1.copy_(Ht.detach())
        self.Ct_1.copy_(Ct.detach())

        agent_index = world.agents.index(agent)
        final_embedding = Ht[agent_index].detach()

        embedding_np = final_embedding.cpu().numpy().astype(np.float32)

        return embedding_np

