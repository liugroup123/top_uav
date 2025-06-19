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
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)

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
        """获取观察，优化GPU使用和计算效率"""
        obs = []

        # 优化1：一次性转换到GPU，减少数据传输
        with torch.cuda.amp.autocast():  # 使用混合精度加速
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

            # GAT前向传播
            if self.training:
                gat_features = self.gat_network(
                    uav_features,
                    target_features,
                    self.uav_adj,
                    self.target_adj
                )
            else:
                with torch.no_grad():
                    gat_features = self.gat_network(
                        uav_features,
                        target_features,
                        self.uav_adj,
                        self.target_adj
                    )
        
        # 优化2：批量处理GAT特征
        gat_features_np = gat_features.detach().cpu().numpy()

        # 优化3：预计算相对目标位置
        relative_targets = self.target_pos[None, :, :] - self.agent_pos[:, None, :]
        relative_targets = relative_targets.reshape(self.num_agents, -1)

        # 为每个UAV生成观察
        for i in range(self.num_agents):
            if i in self.active_agents:
                obs_i = np.concatenate([
                    self.agent_pos[i],          # 位置
                    self.agent_vel[i],          # 速度
                    relative_targets[i],        # 预计算的相对目标位置
                    self._get_neighbor_info(i), # 邻居信息
                    gat_features_np[i],        # GAT特征
                    self._get_topology_info()   # 拓扑变化信息
                ])
                obs.append(obs_i.astype(np.float32))
            else:
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
        """优化的邻接矩阵构建"""
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
# 