import numpy as np
import gym
from gym import spaces
import pygame
import os
from gym.utils import seeding
import torch
from torch_geometric.data import Data
from .gat_model_top import UAVAttentionNetwork, create_adjacency_matrices
from .config import ExperimentConfig, create_config, config_manager
import pdb

class UAVEnv(gym.Env):
    def __init__(
        self,
        config=None,  # 配置对象或配置名称
        config_file=None,  # 配置文件路径
        # 以下参数用于向后兼容，如果提供config则忽略
        num_agents=None,
        num_targets=None,
        world_size=None,
        coverage_radius=None,
        communication_radius=None,
        max_steps=None,
        render_mode=None,
        screen_size=None,
        render_fps=None,
        dt=None,
        experiment_type=None,
        topology_change_interval=None,
        topology_change_probability=None,
        min_active_agents=None,
        max_active_agents=None,
        initial_active_ratio=None
    ):
        super().__init__()

        # 处理配置
        self.config = self._load_config(
            config, config_file,
            num_agents, num_targets, world_size, coverage_radius,
            communication_radius, max_steps, render_mode, screen_size,
            render_fps, dt, experiment_type, topology_change_interval,
            topology_change_probability, min_active_agents, max_active_agents,
            initial_active_ratio
        )

        # 从配置中设置环境参数
        env_config = self.config.environment
        self.num_agents = env_config.num_agents
        self.num_targets = env_config.num_targets
        self.world_size = env_config.world_size
        self.coverage_radius = env_config.coverage_radius
        self.communication_radius = env_config.communication_radius
        self.max_steps = env_config.max_steps
        self.dt = env_config.dt
        self.curr_step = 0
        self.max_coverage_rate = 0.0

        # 从物理配置设置参数
        physics_config = self.config.physics
        self.max_accel = physics_config.max_accel
        self.max_speed = physics_config.max_speed

        # 实验类型配置
        topology_config = self.config.topology
        self.experiment_type = topology_config.experiment_type
        self._validate_experiment_type()

        # 保存拓扑参数
        self.custom_topology_params = {
            'change_interval': topology_config.topology_change_interval,
            'change_probability': topology_config.topology_change_probability,
            'min_agents': topology_config.min_active_agents,
            'max_agents': topology_config.max_active_agents,
            'initial_active_ratio': topology_config.initial_active_ratio
        }

        # 拓扑变化配置
        self.topology_config = self._init_topology_config()

        # 渲染配置
        env_config = self.config.environment
        self.render_mode = env_config.render_mode
        self.width, self.height = env_config.screen_size
        self.screen = None
        self.clock = None
        self.metadata = {"render_fps": env_config.render_fps}

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
            32 +                    # GAT特征
            3                       # 拓扑信息
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
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 根据实验类型初始化UAV状态
        self._init_uav_states()
        
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

        self.training = True  # 默认为训练模式
        
        # GAT网络初始化时设置训练模式
        self.gat_network.train()  # 设置为训练模式
        
        # 优化1：添加缓存
        self._cache = {}
        
        # 优化2：预分配常用数组
        self._agent_mask = np.zeros(num_agents, dtype=bool)
        
        # 优化3：设置CUDA优化参数
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        self.reset()

    def reset(self, seed=None):
        self.curr_step = 0
        self.max_coverage_rate = 0.0

        # 初始化随机种子
        if seed is not None:
            np.random.seed(seed)

        # 初始化无人机位置（确保使用float32）
        self.agent_pos = []
        bottom_y = -self.world_size + 0.15
        spacing = 2 * self.world_size / (self.num_agents + 1)
        for i in range(self.num_agents):
            x = -self.world_size + (i + 1) * spacing
            self.agent_pos.append([x, bottom_y])
        self.agent_pos = np.array(self.agent_pos, dtype=np.float32)  # 明确指定float32

        # 初始化速度
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 初始化目标点位置
        self.target_pos = np.random.uniform(
            -self.world_size*0.85, 
            self.world_size*0.85, 
            (self.num_targets, 2)
        ).astype(np.float32)  # 确保使用float32

        # 构建初始邻接矩阵
        self._build_adjacency_matrices()

        # 重置历史记录
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)

        obs_list = self._get_obs()
        return {f"agent_{i}": obs_list[i] for i in range(self.num_agents)}, {}


    def step(self, action_dict):
        """优化的步进函数"""
        # 优化1：预分配动作数组
        actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        
        # 优化2：批量处理动作
        for i in self.active_agents:
            actions[i] = action_dict[f"agent_{i}"]
        
        # 优化3：批量更新状态
        mask = np.array([i in self.active_agents for i in range(self.num_agents)])
        
        # 处理动作和更新状态
        self.agent_vel[mask] = np.array([
            self._process_action_and_dynamics(actions[i], i)
            for i in self.active_agents
        ])
        
        # 更新位置
        self.agent_pos[mask] += self.agent_vel[mask] * self.dt
        np.clip(self.agent_pos, -self.world_size, self.world_size, out=self.agent_pos)
        
        # 非活跃UAV速度置零
        self.agent_vel[~mask] = 0
        
        # 更新邻接矩阵
        self._build_adjacency_matrices()

        self.curr_step += 1

        # 根据实验类型触发拓扑变化
        self.trigger_topology_change()

        obs_list = self._get_obs()
        rewards_list = self._compute_rewards()
        dones_list = [self.curr_step >= self.max_steps] * self.num_agents

        return (
            {f"agent_{i}": obs_list[i] for i in range(self.num_agents)},
            {f"agent_{i}": rewards_list[i] for i in range(self.num_agents)},
            {f"agent_{i}": dones_list[i] for i in range(self.num_agents)},
            False,
            {f"agent_{i}": {} for i in range(self.num_agents)}
        )


    """
    观察函数相关的函数
    """
    def _get_obs(self):
        """获取观察，处理动态UAV数量"""
        obs = []

        # 创建活跃UAV的索引映射表
        active_idx_map = {agent_idx: i for i, agent_idx in enumerate(self.active_agents)}
        
        # 基础特征：位置和速度
        active_positions = self.agent_pos[self.active_agents]
        active_velocities = self.agent_vel[self.active_agents]
        
        uav_features = torch.tensor(
            np.concatenate([active_positions, active_velocities], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        target_features = torch.tensor(
            self.target_pos,
            dtype=torch.float32,
            device=self.device
        )

        # 获取活跃UAV的邻接矩阵
        active_uav_adj = self._get_active_adj_matrix()
        active_target_adj = self._get_active_target_adj_matrix()

        # GAT前向传播
        if self.training:
            gat_features = self.gat_network(
                uav_features,
                target_features,
                active_uav_adj,
                active_target_adj
            )
        else:
            with torch.no_grad():
                gat_features = self.gat_network(
                    uav_features,
                    target_features,
                    active_uav_adj,
                    active_target_adj
                )

        gat_features_np = gat_features.detach().cpu().numpy()

        # 为每个UAV生成观察
        for i in range(self.num_agents):
            if i in self.active_agents:
                # 使用映射表获取正确的GAT特征索引
                gat_idx = active_idx_map[i]
                
                obs_i = np.concatenate([
                    self.agent_pos[i],                # 位置 (2)
                    self.agent_vel[i],                # 速度 (2)
                    self._get_relative_targets(i),    # 目标相对位置 (num_targets*2)
                    self._get_neighbor_info(i),       # 邻居信息 (num_agents-1)*2
                    gat_features_np[gat_idx],         # GAT特征 (hidden_size//2)
                    self._get_topology_info()         # 拓扑信息 (3)
                ])
                obs.append(obs_i.astype(np.float32))
            else:
                # 对于非活跃UAV，返回零向量
                obs.append(np.zeros(self.observation_space[0].shape, dtype=np.float32))

        return obs

    def _get_relative_targets(self, agent_idx):
        """获取相对目标位置"""
        return (self.target_pos - self.agent_pos[agent_idx]).reshape(-1).tolist()

    def _get_neighbor_info(self, agent_idx):
        """获取邻居信息"""
        neighbor_info = []
        for j in range(self.num_agents):
            if j != agent_idx:
                if j in self.active_agents:
                    delta = self.agent_pos[j] - self.agent_pos[agent_idx]
                    if np.linalg.norm(delta) <= self.communication_radius:
                        neighbor_info.extend(delta)
                    else:
                        neighbor_info.extend([0.0, 0.0])
                else:
                    neighbor_info.extend([0.0, 0.0])
        return neighbor_info

    def _get_topology_info(self):
        """获取拓扑变化信息"""
        return [
            1.0 if self.topology_change['in_progress'] else 0.0,
            1.0 if self.topology_change['change_type'] == 'failure' else 0.0,
            float(self.curr_step - self.topology_change['start_step']) / self.max_steps 
            if self.topology_change['in_progress'] else 0.0
        ]
    
    # 构建邻接矩阵
    def _build_adjacency_matrices(self):
        """构建完整的邻接矩阵（用于可视化）"""
        # 确保使用float32类型
        uav_positions = torch.tensor(self.agent_pos, dtype=torch.float32, device=self.device)
        target_positions = torch.tensor(self.target_pos, dtype=torch.float32, device=self.device)
        
        # 批量计算距离
        uav_dists = torch.cdist(uav_positions, uav_positions)
        target_dists = torch.cdist(uav_positions, target_positions)
        
        # 使用向量化操作
        self.uav_adj = (uav_dists <= self.communication_radius).float()
        self.target_adj = (target_dists <= self.coverage_radius).float()
        
        # 移除自环
        self.uav_adj.fill_diagonal_(0)
        
        # 非活跃UAV的连接设为0
        mask = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        for i in self.active_agents:
            mask[i] = True
        
        # 将非活跃UAV的连接设为0
        self.uav_adj[~mask, :] = 0
        self.uav_adj[:, ~mask] = 0
        self.target_adj[~mask, :] = 0

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
            pre_change_coverage = self.topology_change['pre_change_coverage']

            # 避免除零错误
            if pre_change_coverage is None or pre_change_coverage <= 0:
                # 如果之前覆盖率为0，则直接使用当前覆盖率作为奖励
                recovery_reward = self.reward_params['reorganization_weight'] * current_coverage
            else:
                coverage_recovery = current_coverage / pre_change_coverage
                recovery_reward = self.reward_params['reorganization_weight'] * coverage_recovery

            return recovery_reward

        return 0.0

    def _compute_rewards(self):
        """修改后的奖励计算，只考虑活跃UAV"""
        coverage_reward = self._compute_coverage_reward()
        connectivity_reward = self._compute_connectivity_reward()
        stability_reward = self._compute_stability_reward()
        reorganization_reward = self._compute_reorganization_reward()
        
        rewards = []
        for i in range(self.num_agents):
            if i in self.active_agents:
                energy_reward = self._compute_energy_efficiency_reward(i, self.prev_actions[i])
                boundary_reward = self._compute_boundary_penalty(i)
                
                total_reward = (
                    coverage_reward +
                    connectivity_reward +
                    stability_reward +
                    energy_reward +
                    boundary_reward +
                    reorganization_reward
                )
            else:
                total_reward = 0.0  # 非活跃UAV没有奖励
            
            rewards.append(total_reward)
        
        return rewards

    def calculate_coverage_complete(self):
        """修改后的覆盖率计算，只考虑活跃UAV"""
        covered_flags = []
        for target in self.target_pos:
            covered = False
            for agent_idx in self.active_agents:  # 只考虑活跃UAV
                distance = np.linalg.norm(target - self.agent_pos[agent_idx])
                if distance <= self.coverage_radius:
                    covered = True
                    break
            covered_flags.append(covered)

        covered_count = sum(covered_flags)
        total_targets = len(self.target_pos)
        coverage_rate = covered_count / total_targets if total_targets > 0 else 0

        # 只考虑活跃UAV的连通性
        active_num = len(self.active_agents)
        adjacency_matrix = np.zeros((active_num, active_num), dtype=bool)
        for i, agent_i in enumerate(self.active_agents):
            for j, agent_j in enumerate(self.active_agents[i + 1:], i + 1):
                distance = np.linalg.norm(self.agent_pos[agent_i] - self.agent_pos[agent_j])
                if distance <= self.communication_radius:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True

        visited = [False] * active_num
        def dfs(idx):
            visited[idx] = True
            for j, connected in enumerate(adjacency_matrix[idx]):
                if connected and not visited[j]:
                    dfs(j)

        if active_num > 0:  # 确保有活跃UAV
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

        # 画目标点（绿色）
        for tpos in self.target_pos:
            sx, sy = to_screen(tpos)
            pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 5)

        # 存储无人机屏幕位置用于连线
        screen_positions = {}

        # 绘制所有UAV
        for i, apos in enumerate(self.agent_pos):
            sx, sy = to_screen(apos)
            screen_positions[i] = (sx, sy)

            # 绘制探测半径圆圈（实线）
            coverage_radius_px = int((self.coverage_radius / fixed_cam) * (self.width/2))
            pygame.draw.circle(self.screen, (0, 0, 255), (sx, sy), coverage_radius_px, 1)
            
            # 绘制通信半径圆圈（蓝色虚线）
            comm_radius_px = int((self.communication_radius / fixed_cam) * (self.width/2))
            # 创建虚线效果
            num_segments = 80
            for seg in range(num_segments):
                if seg % 2 == 0:
                    start_angle = 2 * np.pi * seg / num_segments
                    end_angle = 2 * np.pi * (seg + 1) / num_segments
                    start_pos = (
                        sx + int(comm_radius_px * np.cos(start_angle)),
                        sy + int(comm_radius_px * np.sin(start_angle))
                    )
                    end_pos = (
                        sx + int(comm_radius_px * np.cos(end_angle)),
                        sy + int(comm_radius_px * np.sin(end_angle))
                    )
                    pygame.draw.line(self.screen, (70, 130, 180), start_pos, end_pos, 1)

            # 只有UAV中心点的颜色根据状态改变
            if i in self.active_agents:
                color = (0, 0, 255)  # 活跃UAV用蓝色
            else:
                color = (128, 128, 128)  # 失效UAV用灰色
            
            # 绘制UAV本体
            pygame.draw.circle(self.screen, color, (sx, sy), 8)

        # 只在活跃UAV之间绘制红色连接线
        for i in range(self.num_agents):
            if i not in self.active_agents:
                continue  # 跳过非活跃UAV
            
            for j in range(i + 1, self.num_agents):
                if j not in self.active_agents:
                    continue  # 跳过非活跃UAV
                
                # 只在活跃UAV之间检查连接
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist <= self.communication_radius:
                    pygame.draw.line(
                        self.screen, 
                        (255, 0, 0),  # 红色连接线
                        screen_positions[i], 
                        screen_positions[j], 
                        1  # 线宽
                    )

        # 显示实验类型信息
        font = pygame.font.Font(None, 24)
        experiment_text = f"实验类型: {self.experiment_type}"
        text_surface = font.render(experiment_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # 显示活跃UAV数量
        active_count_text = f"活跃UAV: {len(self.active_agents)}/{self.num_agents}"
        text_surface = font.render(active_count_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 35))

        # 显示拓扑变化状态
        if self.topology_change['in_progress']:
            font = pygame.font.Font(None, 36)
            if self.topology_change['change_type'] == 'failure':
                text = f"UAV {self.topology_change['affected_agent']} Failed"
                color = (255, 0, 0)  # 红色
            else:
                text = f"New UAV {self.topology_change['affected_agent']} Added"
                color = (0, 255, 0)  # 绿色

            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect()
            text_rect.topright = (self.width - 10, 10)
            self.screen.blit(text_surface, text_rect)

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

    def train(self):
        """设置为训练模式，并进行相关优化"""
        self.training = True
        self.gat_network.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理GPU缓存

    def eval(self):
        """设置为评估模式，并进行相关优化"""
        self.training = False
        self.gat_network.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理GPU缓存

    def random_fail_uav(self):
        """随机选择一个活跃的UAV进行失效"""
        if len(self.active_agents) > 1:  # 确保至少保留一个UAV
            fail_idx = np.random.choice(self.active_agents)
            return self.fail_uav(fail_idx)
        return False

    def _get_active_adj_matrix(self):
        """获取只包含活跃UAV的邻接矩阵"""
        active_positions = torch.tensor(
            self.agent_pos[self.active_agents],
            dtype=torch.float32,
            device=self.device
        )
        dists = torch.cdist(active_positions, active_positions)
        adj_matrix = (dists <= self.communication_radius).float()
        adj_matrix.fill_diagonal_(0)
        return adj_matrix

    def _get_active_target_adj_matrix(self):
        """获取活跃UAV与目标之间的邻接矩阵"""
        active_positions = torch.tensor(
            self.agent_pos[self.active_agents],
            dtype=torch.float32,
            device=self.device
        )
        target_positions = torch.tensor(
            self.target_pos,
            dtype=torch.float32,
            device=self.device
        )
        
        # 计算活跃UAV与目标之间的距离
        dists = torch.cdist(active_positions, target_positions)
        adj_matrix = (dists <= self.coverage_radius).float()
        return adj_matrix

    def _validate_experiment_type(self):
        """验证实验类型参数"""
        valid_types = ['normal', 'uav_loss', 'uav_addition', 'random_mixed']
        if self.experiment_type not in valid_types:
            raise ValueError(f"实验类型必须是以下之一: {valid_types}, 当前值: {self.experiment_type}")

    def _init_uav_states(self):
        """根据实验类型初始化UAV状态"""
        # 基础UAV状态初始化
        self.uav_states = {
            i: {
                'active': True,           # 初始都是活跃的
                'last_position': None,    # 失效时的位置
                'last_velocity': None,    # 失效时的速度
                'failure_step': None      # 失效的时间步
            } for i in range(self.num_agents)
        }

        if self.experiment_type == 'uav_addition':
            # UAV添加模式：开始时只激活部分UAV，为后续添加留出空间
            if self.custom_topology_params['initial_active_ratio'] is not None:
                # 使用用户指定的比例
                ratio = max(0.3, min(0.8, self.custom_topology_params['initial_active_ratio']))  # 限制在30%-80%
                initial_active_count = max(3, int(self.num_agents * ratio))
            else:
                # 使用默认逻辑
                initial_active_count = max(3, self.num_agents - 2)  # 至少3个，最多num_agents-2个

            self.active_agents = list(range(initial_active_count))

            # 将其余UAV设为非活跃状态
            for i in range(initial_active_count, self.num_agents):
                self.uav_states[i]['active'] = False
        else:
            # 其他模式：初始时所有UAV都是活跃的
            self.active_agents = list(range(self.num_agents))

    def _init_topology_config(self):
        """初始化拓扑变化配置"""
        # 默认配置
        config = {
            'enabled': self.experiment_type != 'normal',
            'change_interval': 50,  # 拓扑变化的间隔步数
            'change_probability': 0.02,  # 随机模式下每步的变化概率
            'min_agents': 3,  # 最少保持的UAV数量
            'max_agents': self.num_agents,  # 最多UAV数量
            'last_change_step': 0,  # 上次变化的步数
        }

        # 根据实验类型调整默认配置
        if self.experiment_type == 'uav_loss':
            config['change_type'] = 'loss_only'
            config['change_interval'] = 80  # 损失模式间隔更长
        elif self.experiment_type == 'uav_addition':
            config['change_type'] = 'addition_only'
            config['change_interval'] = 60  # 添加模式间隔适中
        elif self.experiment_type == 'random_mixed':
            config['change_type'] = 'random'
            config['change_probability'] = 0.015  # 随机模式概率稍低

        # 应用用户自定义参数（如果提供）
        if self.custom_topology_params['change_interval'] is not None:
            config['change_interval'] = self.custom_topology_params['change_interval']

        if self.custom_topology_params['change_probability'] is not None:
            config['change_probability'] = self.custom_topology_params['change_probability']

        if self.custom_topology_params['min_agents'] is not None:
            config['min_agents'] = max(1, self.custom_topology_params['min_agents'])  # 至少1个

        if self.custom_topology_params['max_agents'] is not None:
            config['max_agents'] = min(self.num_agents, self.custom_topology_params['max_agents'])

        return config

    def trigger_topology_change(self):
        """根据实验类型触发拓扑变化"""
        if not self.topology_config['enabled']:
            return False

        # 检查是否应该触发变化
        should_change = False

        if self.experiment_type == 'random_mixed':
            # 随机模式：基于概率触发
            should_change = np.random.random() < self.topology_config['change_probability']
        else:
            # 固定间隔模式
            steps_since_last = self.curr_step - self.topology_config['last_change_step']
            should_change = steps_since_last >= self.topology_config['change_interval']

        if not should_change:
            return False

        # 执行拓扑变化
        change_executed = False

        if self.experiment_type == 'uav_loss':
            change_executed = self._execute_uav_loss()
        elif self.experiment_type == 'uav_addition':
            change_executed = self._execute_uav_addition()
        elif self.experiment_type == 'random_mixed':
            change_executed = self._execute_random_change()

        if change_executed:
            self.topology_config['last_change_step'] = self.curr_step

        return change_executed

    def _execute_uav_loss(self):
        """执行UAV损失"""
        if len(self.active_agents) <= self.topology_config['min_agents']:
            return False

        # 随机选择一个UAV失效
        fail_idx = np.random.choice(self.active_agents)
        success = self.fail_uav(fail_idx)

        if success:
            print(f"[实验模式: UAV损失] Step {self.curr_step}: UAV {fail_idx} 失效")

        return success

    def _execute_uav_addition(self):
        """执行UAV添加"""
        if len(self.active_agents) >= self.topology_config['max_agents']:
            return False

        # 添加新UAV
        new_uav_idx = self.add_uav()

        if new_uav_idx is not None:
            print(f"[实验模式: UAV添加] Step {self.curr_step}: 添加新UAV {new_uav_idx}")
            return True

        return False

    def _execute_random_change(self):
        """执行随机变化（损失或添加）"""
        # 50%概率选择损失或添加
        if np.random.random() < 0.5:
            # 尝试UAV损失
            if len(self.active_agents) > self.topology_config['min_agents']:
                return self._execute_uav_loss()
        else:
            # 尝试UAV添加
            if len(self.active_agents) < self.topology_config['max_agents']:
                return self._execute_uav_addition()

        return False

    def get_experiment_info(self):
        """获取实验信息"""
        return {
            'experiment_type': self.experiment_type,
            'topology_enabled': self.topology_config['enabled'],
            'active_agents_count': len(self.active_agents),
            'total_agents': self.num_agents,
            'current_step': self.curr_step,
            'last_change_step': self.topology_config['last_change_step'],
            'topology_in_progress': self.topology_change['in_progress'],
            'change_type': self.topology_change['change_type']
        }

    def set_experiment_type(self, experiment_type):
        """动态设置实验类型"""
        old_type = self.experiment_type
        self.experiment_type = experiment_type
        self._validate_experiment_type()
        self.topology_config = self._init_topology_config()
        print(f"实验类型已从 '{old_type}' 更改为 '{experiment_type}'")

    def _load_config(self, config, config_file, *args):
        """加载配置"""
        from .config import ExperimentConfig, create_config, config_manager

        # 如果提供了配置对象
        if isinstance(config, ExperimentConfig):
            return config

        # 如果提供了配置文件路径
        if config_file is not None:
            return config_manager.load_config(config_file)

        # 如果提供了配置名称
        if isinstance(config, str):
            return create_config(config)

        # 如果没有提供配置，使用传入的参数创建配置
        if config is None:
            (num_agents, num_targets, world_size, coverage_radius,
             communication_radius, max_steps, render_mode, screen_size,
             render_fps, dt, experiment_type, topology_change_interval,
             topology_change_probability, min_active_agents, max_active_agents,
             initial_active_ratio) = args

            from .config import EnvironmentConfig, TopologyConfig, RewardConfig, PhysicsConfig, GATConfig

            # 创建默认配置并应用传入的参数
            env_config = EnvironmentConfig()
            if num_agents is not None: env_config.num_agents = num_agents
            if num_targets is not None: env_config.num_targets = num_targets
            if world_size is not None: env_config.world_size = world_size
            if coverage_radius is not None: env_config.coverage_radius = coverage_radius
            if communication_radius is not None: env_config.communication_radius = communication_radius
            if max_steps is not None: env_config.max_steps = max_steps
            if render_mode is not None: env_config.render_mode = render_mode
            if screen_size is not None: env_config.screen_size = screen_size
            if render_fps is not None: env_config.render_fps = render_fps
            if dt is not None: env_config.dt = dt

            topology_config = TopologyConfig()
            if experiment_type is not None: topology_config.experiment_type = experiment_type
            if topology_change_interval is not None: topology_config.topology_change_interval = topology_change_interval
            if topology_change_probability is not None: topology_config.topology_change_probability = topology_change_probability
            if min_active_agents is not None: topology_config.min_active_agents = min_active_agents
            if max_active_agents is not None: topology_config.max_active_agents = max_active_agents
            if initial_active_ratio is not None: topology_config.initial_active_ratio = initial_active_ratio

            return ExperimentConfig(
                name="runtime_config",
                description="运行时创建的配置",
                environment=env_config,
                topology=topology_config,
                reward=RewardConfig(),
                physics=PhysicsConfig(),
                gat=GATConfig()
            )

        raise ValueError("必须提供config、config_file或具体参数")

    def save_current_config(self, filename: str):
        """保存当前配置到文件"""
        config_manager.save_config(self.config, filename)

    def load_config_from_file(self, filename: str):
        """从文件重新加载配置并重新初始化环境"""
        new_config = config_manager.load_config(filename)
        self.__init__(config=new_config)

    # ========== GAT训练相关方法 ==========

    def get_gat_parameters(self):
        """获取GAT网络的参数，用于外部训练"""
        return self.gat_network.parameters()

    def get_gat_named_parameters(self):
        """获取GAT网络的命名参数"""
        return self.gat_network.named_parameters()

    def get_gat_state_dict(self):
        """获取GAT网络的状态字典"""
        return self.gat_network.state_dict()

    def load_gat_state_dict(self, state_dict):
        """加载GAT网络的状态字典"""
        self.gat_network.load_state_dict(state_dict)

    def save_gat_model(self, filepath):
        """保存GAT模型"""
        torch.save(self.gat_network.state_dict(), filepath)
        print(f"GAT模型已保存到: {filepath}")

    def load_gat_model(self, filepath):
        """加载GAT模型"""
        self.gat_network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"GAT模型已从 {filepath} 加载")

    def get_gat_features_with_grad(self, return_loss=False):
        """
        获取带梯度的GAT特征，用于端到端训练

        Args:
            return_loss: 是否返回GAT的内部损失

        Returns:
            gat_features: 带梯度的GAT特征
            loss: GAT内部损失（如果return_loss=True）
        """
        # 基础特征：位置和速度
        active_positions = self.agent_pos[self.active_agents]
        active_velocities = self.agent_vel[self.active_agents]

        uav_features = torch.tensor(
            np.concatenate([active_positions, active_velocities], axis=1),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True  # 启用梯度
        )
        target_features = torch.tensor(
            self.target_pos,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True  # 启用梯度
        )

        # 获取活跃UAV的邻接矩阵
        active_uav_adj = self._get_active_adj_matrix()
        active_target_adj = self._get_active_target_adj_matrix()

        # GAT前向传播（保持梯度）
        gat_features = self.gat_network(
            uav_features,
            target_features,
            active_uav_adj,
            active_target_adj
        )

        if return_loss:
            # 计算GAT的内部损失（可选）
            # 这里可以添加GAT特定的损失，比如注意力正则化
            attention_loss = 0.0  # 占位符
            return gat_features, attention_loss

        return gat_features

    def update_gat_with_loss(self, loss, optimizer):
        """
        使用给定的损失更新GAT参数

        Args:
            loss: 包含GAT梯度的损失
            optimizer: GAT的优化器
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def get_gat_info(self):
        """获取GAT网络信息"""
        total_params = sum(p.numel() for p in self.gat_network.parameters())
        trainable_params = sum(p.numel() for p in self.gat_network.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'training_mode': self.gat_network.training,
            'model_structure': str(self.gat_network)
        }
#