import numpy as np
import gym
from gym import spaces
import pygame
import os
from gym.utils import seeding
import torch
from torch_geometric.data import Data
from mpe_uav.uav_env.gat_model import UAVAttentionNetwork, create_adjacency_matrices
import pdb

class UAVEnv(gym.Env):
    def __init__(
        self,
        num_agents=5,
        num_targets=10,
        world_size=1.0,
        coverage_radius=0.3,
        communication_radius=0.6,
        max_steps=200,
        render_mode=None,
        screen_size=(700, 700),
        render_fps=60,
        dt=0.1
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.world_size = world_size
        self.coverage_radius = coverage_radius
        self.communication_radius = communication_radius
        self.max_steps = max_steps
        self.curr_step = 0
        self.dt = dt  
        self.max_coverage_rate = 0.0
        self.max_accel=1.5
        self.max_speed=2.0

        # 渲染配置
        self.render_mode = render_mode
        self.width, self.height = screen_size
        self.screen = None
        self.clock = None
        self.metadata = {"render_fps": render_fps}

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GAT初始化（传入device参数）
        self.gat_network = UAVAttentionNetwork(
            uav_features=4,  # 位置(2) + 速度(2)
            target_features=2,  # 目标位置(2)
            hidden_size=64,
            heads=4,
            dropout=0.6,
            device=self.device  # 添加device参数
    )
        # 将网络移到指定设备
        self.gat_network = self.gat_network.to(self.device)

        # 物理参数配置
        self.physics_params = {
            'max_accel': 3.0,
            'max_speed': 5.0,
            'boundary_decay_zone': 0.2,
            'min_speed_ratio': 0.2,
            'bounce_energy_loss': 0.7
        }
        
        # 动作空间配置
        self.action_scale = 1.0  # 动作缩放因子
        self.action_space = [
            spaces.Box(
                low=-self.action_scale,
                high=self.action_scale,
                shape=(2,),
                dtype=np.float32
            ) for _ in range(self.num_agents)
        ]
        # 观察空间
        obs_dim = (
            4 +                     # 基础状态 (位置+速度)
            2 * self.num_targets + # 目标相对位置
            2 * (self.num_agents - 1) + # 邻居信息
            32                     # GAT特征
        )
        self.observation_space = [spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        ) for _ in range(self.num_agents)]

        # 添加奖励函数配置参数
        self.reward_params = {
            # 覆盖率相关
            'coverage_weight': 3.5,          # 从35.0降到3.5
            'coverage_exp': 1.5,             # 保持不变
            'distance_scale': 1.5,           # 从15.0降到1.5
            
            # 连通性相关
            'connectivity_weight': 2.0,      # 从20.0降到2.0
            'min_connectivity_ratio': 0.3,    # 保持不变
            
            # 稳定性相关
            'stability_weight': 1.5,         # 从15.0降到1.5
            'stability_time_window': 10,      # 保持不变
            'stability_threshold': 0.9,       # 保持不变
            
            # 能量效率相关
            'energy_weight': 0.5,            # 从5.0降到0.5
            'smoothness_weight': 0.5,        # 从5.0降到0.5
            
            # 边界惩罚相关
            'boundary_weight': 1.0,          # 从10.0降到1.0
            'safe_distance': 0.1             # 保持不变
        }
        
        # 添加历史记录用于计算稳定性
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2))

        # 初始时所有UAV都是活跃的
        self.active_agents = list(range(num_agents))
        self.uav_states = {
            i: {
                'active': True,           # 初始都是活跃的
                'last_position': None,    # 失效时的位置
                'last_velocity': None,    # 失效时的速度
                'failure_step': None      # 失效的时间步
            } for i in range(num_agents)
        }
        
        # 添加拓扑变化相关的奖励参数
        self.reward_params.update({
            'reorganization_weight': 2.0,    # 重组行为的奖励权重
            'task_reassign_weight': 1.5,     # 任务重分配的奖励权重
            'formation_weight': 1.0,         # 队形维持的奖励权重
        })
        
        # 拓扑变化状态
        self.topology_change = {
            'in_progress': False,            # 是否正在发生拓扑变化
            'change_type': None,             # 'failure' 或 'addition'
            'affected_agent': None,          # 受影响的UAV索引
            'start_step': None,              # 变化开始的步数
            'pre_change_coverage': None,     # 变化前的覆盖率
        }

        self.reset()

    def reset(self, seed=None):
        self.curr_step = 0
        self.max_coverage_rate = 0.0


        # 初始化随机种子（确保复现性）
        if seed is not None:
            np.random.seed(seed)

        # 初始化无人机位置（靠底部排列，防止重叠）
        self.agent_pos = []
        bottom_y = -self.world_size + 0.15
        spacing = 2 * self.world_size / (self.num_agents + 1)
        for i in range(self.num_agents):
            x = -self.world_size + (i + 1) * spacing
            self.agent_pos.append([x, bottom_y])
        self.agent_pos = np.array(self.agent_pos, dtype=np.float32)

        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 初始化目标点位置（随机散布）
        self.target_pos = np.random.uniform(-self.world_size*0.85, self.world_size*0.85, (self.num_targets, 2))

        # 构建初始邻接矩阵
        self._build_adjacency_matrices()

        # 重置历史记录
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2))

        obs_list = self._get_obs()
        return {f"agent_{i}": obs_list[i] for i in range(self.num_agents)},{}


    def step(self, action_dict):
        actions = []
        for i in range(self.num_agents):
            if i in self.active_agents:
                # 活跃的UAV使用正常动作
                actions.append(action_dict[f"agent_{i}"])
            else:
                # 失效的UAV保持在原地，速度为0
                actions.append(np.zeros(self.action_space[0].shape))
        
        # 处理每个智能体的动作
        for i, action in enumerate(actions):
            if i in self.active_agents:
                # 只更新活跃UAV的状态
                self.agent_vel[i] = self._process_action_and_dynamics(action, i)
                self.agent_pos[i] += self.agent_vel[i] * self.dt
                self.agent_pos[i] = np.clip(self.agent_pos[i], -self.world_size, self.world_size)
            else:
                # 失效的UAV保持在最后的位置，速度为0
                self.agent_vel[i] = np.zeros_like(self.agent_vel[i])
        
        # 更新邻接矩阵
        self._build_adjacency_matrices()

        self.curr_step += 1 # 步数加1
        obs_list = self._get_obs()  #获取当前观察
        rewards_list = self._compute_rewards()  #计算奖励函数
        dones_list = [self.curr_step >= self.max_steps] * self.num_agents

        obs = {f"agent_{i}": obs_list[i] for i in range(self.num_agents)}
        rewards = {f"agent_{i}": rewards_list[i] for i in range(self.num_agents)}
        dones = {f"agent_{i}": dones_list[i] for i in range(self.num_agents)}
        infos = {f"agent_{i}": {} for i in range(self.num_agents)}

        return obs, rewards, dones, False, infos


    """
    观察函数相关的函数
    """
    def _get_obs(self):
        obs = []

        # 基础特征：位置和速度
        uav_features = torch.tensor(
            np.concatenate([self.agent_pos, self.agent_vel], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        target_features = torch.tensor(
            self.target_pos,
            dtype=torch.float32,
            device=self.device
        )

        # GAT前向传播（静态特征提取）
        with torch.no_grad():
            gat_features = self.gat_network(
                uav_features,
                target_features,
                self.uav_adj,
                self.target_adj
            ).detach().cpu().numpy()  # 优化：提前 detach 并移动到 CPU

        # 每个智能体的观察拼接
        for i in range(self.num_agents):
            if i in self.active_agents:  # 只为活跃的UAV生成观察
                o = [*self.agent_pos[i], *self.agent_vel[i]]

                # 相对目标点位置
                rel_targets = (self.target_pos - self.agent_pos[i]).reshape(-1).tolist()
                o += rel_targets

                # 邻居相对位置（带通信判断）
                neigh = []
                for j in range(self.num_agents):
                    if j == i:
                        continue
                    if j in self.active_agents:  # 只考虑活跃的邻居
                        delta = self.agent_pos[j] - self.agent_pos[i]
                        if np.linalg.norm(delta) <= self.communication_radius:
                            neigh += delta.tolist()
                        else:
                            neigh += [0.0, 0.0]
                    else:
                        neigh += [0.0, 0.0]  # 对于非活跃的UAV，添加零向量
                o += neigh

                # 拼接 GAT 特征
                o += gat_features[i].tolist()

                # 添加拓扑变化信息
                topology_info = [
                    1.0 if self.topology_change['in_progress'] else 0.0,
                    1.0 if self.topology_change['change_type'] == 'failure' else 0.0,
                    float(self.curr_step - self.topology_change['start_step']) / self.max_steps if self.topology_change['in_progress'] else 0.0
                ]
                o += topology_info

                obs.append(np.array(o, dtype=np.float32))
            else:
                # 对于非活跃的UAV，生成零观察
                obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

        return obs
    
    # 构建邻接矩阵
    def _build_adjacency_matrices(self):
        """
        构建UAV-UAV和UAV-Target的邻接矩阵
        """
        uav_positions = torch.tensor(self.agent_pos, device=self.device)
        target_positions = torch.tensor(self.target_pos, device=self.device)
        
        self.uav_adj, self.target_adj = create_adjacency_matrices(
            uav_positions=uav_positions,
            target_positions=target_positions,
            comm_radius=self.communication_radius,
            coverage_radius=self.coverage_radius
        )
        

    """
    奖励函数相关的函数
    """
    def _compute_coverage_reward(self):
        """计算覆盖率奖励"""
        coverage_rate, _, _, _ = self.calculate_coverage_complete()
        
        # 计算到目标的平均最小距离
        distances = np.linalg.norm(self.agent_pos[:, None, :] - self.target_pos[None, :, :], axis=2)
        avg_min_dist = np.mean(np.min(distances, axis=0))
        normalized_dist = np.clip(avg_min_dist, 0, self.reward_params['distance_scale'])
        
        # 非线性覆盖率奖励
        coverage_score = (coverage_rate ** self.reward_params['coverage_exp']) * normalized_dist
        return coverage_score * self.reward_params['coverage_weight']

    def _compute_connectivity_reward(self):
        """计算连通性奖励"""
        _, connected, _, unconnected_count = self.calculate_coverage_complete()
        
        # 计算连通率
        connectivity_ratio = 1.0 - (unconnected_count / self.num_agents)
        
        # 使用平滑的连通性奖励
        if connectivity_ratio >= self.reward_params['min_connectivity_ratio']:
            connectivity_reward = self.reward_params['connectivity_weight'] * connectivity_ratio
        else:
            # 低于阈值时给予负奖励
            connectivity_reward = -self.reward_params['connectivity_weight'] * (
                self.reward_params['min_connectivity_ratio'] - connectivity_ratio
            )
            
        return connectivity_reward

    def _compute_stability_reward(self):
        """计算稳定性奖励"""
        coverage_rate, connected, _, _ = self.calculate_coverage_complete()
        self.coverage_history.append(coverage_rate)
        
        # 保持固定长度的历史记录
        window_size = self.reward_params['stability_time_window']
        if len(self.coverage_history) > window_size:
            self.coverage_history = self.coverage_history[-window_size:]
        
        # 计算覆盖率的稳定性
        if len(self.coverage_history) >= 2:
            coverage_variance = np.var(self.coverage_history)
            stability_score = np.exp(-coverage_variance * 10)  # 使用指数函数平滑转换
            
            # 当覆盖率高且稳定时给予额外奖励
            if (coverage_rate > self.reward_params['stability_threshold'] and 
                stability_score > self.reward_params['stability_threshold'] and 
                connected):
                return self.reward_params['stability_weight'] * stability_score
        return 0.0

    def _compute_energy_efficiency_reward(self, agent_idx, action):
        """计算能量效率奖励"""
        # 计算速度变化（加速度）的平方和
        velocity_change = np.sum(np.square(self.agent_vel[agent_idx]))
        energy_penalty = -self.reward_params['energy_weight'] * velocity_change * 0.1  # 添加额外的0.1缩放因子
        
        # 计算动作平滑度
        action_diff = np.sum(np.square(action - self.prev_actions[agent_idx]))
        smoothness_penalty = -self.reward_params['smoothness_weight'] * action_diff * 0.1  # 添加额外的0.1缩放因子
        
        self.prev_actions[agent_idx] = action
        return energy_penalty + smoothness_penalty

    def _compute_boundary_penalty(self, agent_idx):
        """计算边界惩罚"""
        pos = self.agent_pos[agent_idx]
        safe_distance = self.reward_params['safe_distance']
        
        # 计算到边界的距离
        boundary_distances = self.world_size - np.abs(pos)
        
        # 如果距离小于安全距离，给予惩罚
        penalties = np.maximum(0, safe_distance - boundary_distances)
        return -self.reward_params['boundary_weight'] * np.sum(penalties) * 0.1  # 添加额外的0.1缩放因子

    def _compute_reorganization_reward(self):
        """计算重组奖励"""
        if not self.topology_change['in_progress']:
            return 0.0
        
        if self.topology_change['change_type'] == 'failure':
            # 计算失效后的覆盖率恢复程度
            current_coverage = self.calculate_coverage_complete()[0]
            coverage_recovery = current_coverage / self.topology_change['pre_change_coverage']
            
            # 鼓励快速恢复覆盖率
            recovery_reward = self.reward_params['reorganization_weight'] * coverage_recovery
            return recovery_reward
        
        return 0.0

    def _compute_rewards(self):
        """
        计算总奖励
        Returns:
            list: 每个智能体的奖励列表
        """
        # 计算全局组件（所有智能体共享）
        coverage_reward = self._compute_coverage_reward()
        connectivity_reward = self._compute_connectivity_reward()
        stability_reward = self._compute_stability_reward()
        
        # 添加拓扑变化相关的奖励
        reorganization_reward = self._compute_reorganization_reward()
        
        rewards = []
        for i in range(self.num_agents):
            # 计算个体组件
            energy_reward = self._compute_energy_efficiency_reward(i, 
                self.prev_actions[i])
            boundary_reward = self._compute_boundary_penalty(i)
            
            # 合并所有奖励组件
            total_reward = (
                coverage_reward +
                connectivity_reward +
                stability_reward +
                energy_reward +
                boundary_reward +
                reorganization_reward
            )
            
            rewards.append(total_reward)
            
        return rewards

    def calculate_coverage_complete(self):
        covered_flags = []
        for target in self.target_pos:
            covered = False
            for agent in self.agent_pos:
                distance = np.linalg.norm(target - agent)
                if distance <= self.coverage_radius:
                    covered = True
                    break
            covered_flags.append(covered)

        covered_count = sum(covered_flags)
        total_targets = len(self.target_pos)
        coverage_rate = covered_count / total_targets if total_targets > 0 else 0

        num_agents = self.num_agents
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if distance <= self.communication_radius:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        visited = [False] * num_agents
        def dfs(idx):
            visited[idx] = True
            for j, connected in enumerate(adjacency_matrix[idx]):
                if connected and not visited[j]:
                    dfs(j)
        dfs(0)
        fully_connected = all(visited)
        unconnected_count = visited.count(False)

        if fully_connected:
            self.max_coverage_rate = max(self.max_coverage_rate, coverage_rate)

        return coverage_rate, fully_connected, self.max_coverage_rate, unconnected_count
    
    """
    render相关的函数
    """
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "Calling render without specifying render_mode."
            )
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('UAV Topology')
            self.clock = pygame.time.Clock()
        self.draw()
        if self.render_mode == 'rgb_array':
            data = pygame.surfarray.array3d(self.screen)
            return np.transpose(data, (1, 0, 2))
        elif self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def draw(self):
        fixed_cam = self.world_size
        self.screen.fill((255, 255, 255))

        def to_screen(pos):
            x, y = pos
            y = -y  # y 轴翻转
            sx = int((x / fixed_cam) * (self.width / 2) + self.width / 2)
            sy = int((y / fixed_cam) * (self.height / 2) + self.height / 2)
            return sx, sy

        # 画目标点
        for tpos in self.target_pos:
            sx, sy = to_screen(tpos)
            pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 5)

        # 存储无人机屏幕位置用于连线
        screen_positions = []

        for i, apos in enumerate(self.agent_pos):
            sx, sy = to_screen(apos)
            screen_positions.append((sx, sy))

            if i in self.active_agents:
                # 活跃的UAV用蓝色
                color = (0, 0, 255)  # 蓝色
            else:
                # 失效的UAV用灰色
                color = (128, 128, 128)  # 灰色
            
            pygame.draw.circle(self.screen, color, (sx, sy), 8)

            # 绘制探测半径圆圈（实线）
            coverage_radius_px = int((self.coverage_radius / fixed_cam) * (self.width/2))
            pygame.draw.circle(self.screen, (0, 0, 255), (sx, sy), coverage_radius_px, 1)
            
            # 绘制通信半径圆圈（蓝色虚线）
            comm_radius_px = int((self.communication_radius / fixed_cam) * (self.width/2))
            # 创建虚线效果
            num_segments = 80  # 虚线段数
            for i in range(num_segments):
                if i % 2 == 0:  # 只画偶数段，形成虚线
                    start_angle = 2 * np.pi * i / num_segments
                    end_angle = 2 * np.pi * (i + 1) / num_segments
                    # 计算弧段的起点和终点
                    start_pos = (
                        sx + int(comm_radius_px * np.cos(start_angle)),
                        sy + int(comm_radius_px * np.sin(start_angle))
                    )
                    end_pos = (
                        sx + int(comm_radius_px * np.cos(end_angle)),
                        sy + int(comm_radius_px * np.sin(end_angle))
                    )
                    # 画虚线段
                    pygame.draw.line(self.screen, (70, 130, 180), start_pos, end_pos, 1)  # 使用浅蓝色

        # 画红线：若两个无人机之间的距离小于通信范围
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist <= self.communication_radius:
                    pygame.draw.line(self.screen, (255, 0, 0),
                                    screen_positions[i], screen_positions[j], 1)

    """
    计算覆盖率,连通性相关的函数
    """
    def calculate_coverage_complete(self):
        # 计算目标点是否被覆盖
        covered_flags = []
        for target in self.target_pos:
            covered = False
            for agent in self.agent_pos:
                distance = np.linalg.norm(target - agent)
                if distance <= self.coverage_radius:
                    covered = True
                    break
            covered_flags.append(covered)

        # 计算覆盖率
        covered_count = sum(covered_flags)
        total_targets = len(self.target_pos)
        coverage_rate = covered_count / total_targets if total_targets > 0 else 0

        # 构建通信邻接矩阵
        num_agents = self.num_agents
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if distance <= self.communication_radius:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        # DFS 检查连通性
        visited = [False] * num_agents

        def dfs(idx):
            visited[idx] = True
            for neighbor_idx, connected in enumerate(adjacency_matrix[idx]):
                if connected and not visited[neighbor_idx]:
                    dfs(neighbor_idx)

        dfs(0)
        fully_connected = all(visited)
        unconnected_count = visited.count(False)

        # 更新最大覆盖率
        if fully_connected:
            self.max_coverage_rate = max(self.max_coverage_rate, coverage_rate)

        return coverage_rate, fully_connected, self.max_coverage_rate, unconnected_count




    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    
    #返回智能体以及动作空间观察空间
    @property
    def agents(self):
        return [f"agent_{i}" for i in range(self.num_agents)]

    def get_observation_space(self, agent_id):
        return self.observation_space[int(agent_id.split("_")[1])]

    def get_action_space(self, agent_id):
        return self.action_space[int(agent_id.split("_")[1])]

    def _process_action_and_dynamics(self, action, agent_idx):
        """
        统一处理动作归一化和物理约束
        Args:
            action: 原始动作 (-1, 1)范围
            agent_idx: 智能体索引
        Returns:
            更新后的速度
        """
        # 1. 动作归一化到加速度空间
        normalized_accel = action * self.max_accel
        
        # 2. 更新速度
        new_velocity = self.agent_vel[agent_idx] + normalized_accel * self.dt
        
        # 3. 速度限制
        current_speed = np.linalg.norm(new_velocity)
        if current_speed > self.max_speed:
            new_velocity = new_velocity * (self.max_speed / current_speed)
        
        # 4. 边界处理
        pos = self.agent_pos[agent_idx]
        boundary_distance = self.world_size - np.abs(pos)
        
        # 创建边界衰减系数
        decay_zone = 0.15  # 开始减速的边界距离
        min_speed_ratio = 0.2  # 最小速度比例
        
        for i in range(2):  # 对x和y分别处理
            if boundary_distance[i] < decay_zone:
                # # 线性衰减速度
                # decay_ratio = max(
                #     min_speed_ratio,
                #     boundary_distance[i] / decay_zone
                # )
                # new_velocity[i] *= decay_ratio
                
                # 如果已经触碰边界，反向速度
                if abs(pos[i]) >= self.world_size:
                    new_velocity[i] *= -0.85  # 反弹时的能量损失
        
        return new_velocity

    def fail_uav(self, uav_idx):
        """模拟UAV失效"""
        if uav_idx in self.active_agents:
            # 记录失效状态和最后位置
            self.uav_states[uav_idx]['active'] = False
            self.uav_states[uav_idx]['last_position'] = self.agent_pos[uav_idx].copy()
            self.uav_states[uav_idx]['last_velocity'] = np.zeros_like(self.agent_vel[uav_idx])  # 失效后速度为0
            self.uav_states[uav_idx]['failure_step'] = self.curr_step
            
            # 从活跃列表中移除
            self.active_agents.remove(uav_idx)
            print(f"UAV {uav_idx} 已失效，当前位置: {self.uav_states[uav_idx]['last_position']}")
            
            # 记录拓扑变化状态
            self.topology_change = {
                'in_progress': True,
                'change_type': 'failure',
                'affected_agent': uav_idx,
                'start_step': self.curr_step,
                'pre_change_coverage': self.calculate_coverage_complete()[0]
            }
            
            return True
        return False

    def add_uav(self, position=None):
        """添加新的UAV"""
        # 找到第一个非活跃的UAV索引
        for i in range(self.num_agents):
            if i not in self.active_agents:
                # 设置初始位置
                if position is None:
                    position = self._get_safe_spawn_position()
                
                # 激活UAV
                self.agent_pos[i] = np.array(position, dtype=np.float32)
                self.agent_vel[i] = np.zeros(2, dtype=np.float32)
                self.uav_states[i]['active'] = True
                self.active_agents.append(i)
                
                # 记录拓扑变化状态
                self.topology_change = {
                    'in_progress': True,
                    'change_type': 'addition',
                    'affected_agent': i,
                    'start_step': self.curr_step,
                    'pre_change_coverage': self.calculate_coverage_complete()[0]
                }
                
                return i
        return None

    def _get_safe_spawn_position(self):
        """获取安全的生成位置"""
        # 默认在边界生成
        spawn_positions = [
            [-self.world_size, -self.world_size],  # 左下角
            [self.world_size, -self.world_size],   # 右下角
            [0, -self.world_size]                  # 底部中间
        ]
        
        # 选择距离现有UAV最远的位置
        max_min_dist = -1
        best_pos = spawn_positions[0]
        
        for pos in spawn_positions:
            min_dist = float('inf')
            for idx in self.active_agents:
                dist = np.linalg.norm(self.agent_pos[idx] - pos)
                min_dist = min(min_dist, dist)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_pos = pos
            
        return np.array(best_pos)
# 