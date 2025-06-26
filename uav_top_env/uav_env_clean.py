"""
简化的拓扑UAV环境 - 保持原有接口和功能
基于原版uav_env_top.py，删除冗余代码，保留核心功能
"""

import numpy as np
import pygame
import torch
import gym
from gym import spaces
import cv2

# 动态导入GAT模型
import importlib.util
import os

def _import_gat_model():
    """动态导入GAT模型"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gat_model_path = os.path.join(current_dir, 'gat_model_top.py')

    spec = importlib.util.spec_from_file_location("gat_model_top", gat_model_path)
    gat_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gat_module)

    return gat_module.UAVAttentionNetwork, gat_module.create_adjacency_matrices

UAVAttentionNetwork, create_adjacency_matrices = _import_gat_model()

# 简化版本不需要复杂的配置导入


class UAVEnv(gym.Env):
    """简化的拓扑UAV环境 - 保持原有接口"""
    
    def __init__(self,
                 render_mode=None,
                 experiment_type='normal',
                 num_agents=6,
                 num_targets=10,
                 max_steps=200,
                 min_active_agents=3,
                 max_active_agents=None):
        
        # 基础参数设置
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.world_size = 1.0
        self.max_steps = max_steps
        self.experiment_type = experiment_type
        self.render_mode = render_mode
        
        # 物理参数
        self.max_speed = 0.1
        self.communication_range = 0.3
        self.sensing_range = 0.2
        
        # 拓扑参数
        self.min_active_agents = min_active_agents
        self.max_active_agents = max_active_agents or num_agents
        
        # 奖励权重（保持原有设置）
        self.coverage_weight = 3.5
        self.connectivity_weight = 2.0
        self.boundary_weight = 1.0
        self.stability_weight = 1.5
        
        # 概率设置（用于probabilistic模式）
        self.normal_probability = 0.80
        self.loss_probability = 0.15
        self.addition_probability = 0.05
        
        # 状态变量
        self.curr_step = 0
        self.active_agents = list(range(self.num_agents))
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        # 位置和速度
        self.agent_pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.target_pos = np.zeros((self.num_targets, 2), dtype=np.float32)
        
        # Episode计划
        self.episode_plan = {
            'type': 'normal',
            'trigger_step': None,
            'executed': False
        }

        # GAT缓存优化 - 减少CPU-GPU传输
        self.gat_cache = {
            'features': None,
            'last_update_step': -1,
            'update_interval': 3  # 每3步更新一次GAT特征
        }

        # 速度限制参数 - 基于连接性的动态约束
        self.max_base_speed = 0.1           # 基础最大速度
        self.connectivity_speed_factor = 0.5 # 连接性影响因子
        self.min_speed_limit = 0.02         # 最小速度限制
        self.epsilon = 0.05                 # 连接性容忍度参数
        self.speed_violation_penalty = -5.0 # 速度违规惩罚

        # 连接性历史记录
        self.connectivity_history = []
        self.speed_limits_history = []
        
        # GAT网络 - 使用正确的参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gat_model = UAVAttentionNetwork(
            uav_features=4,      # UAV特征维度：位置(2) + 速度(2)
            target_features=2,   # 目标特征维度：位置(2)
            hidden_size=64,      # 隐藏层大小
            heads=4,             # 注意力头数
            dropout=0.1,         # dropout率
            device=self.device
        )
        
        # 训练模式标志
        self.training = False
        
        # 观察和动作空间（保持原有格式）
        self._setup_spaces()
        
        # 渲染相关
        self.screen = None
        self.clock = None
        self.screen_size = 700
        
        # 历史记录
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        
    def _setup_spaces(self):
        """设置观察和动作空间 - 保持原有格式"""
        # 观察维度计算（与原版一致）
        obs_dim = (
            4 +                           # 基础状态 (位置+速度)
            2 * self.num_targets +       # 目标相对位置
            2 * (self.num_agents - 1) +  # 邻居信息
            32 +                         # GAT特征
            3                            # 拓扑信息
        )
        
        # 保持原有的字典格式
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }
    
    def reset(self, seed=None):
        """重置环境 - 保持原有接口"""
        if seed is not None:
            np.random.seed(seed)
        
        self.curr_step = 0
        self.active_agents = list(range(self.num_agents))
        
        # 随机初始化位置
        self.agent_pos = np.random.uniform(-self.world_size, self.world_size, 
                                         (self.num_agents, 2)).astype(np.float32)
        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.target_pos = np.random.uniform(-self.world_size, self.world_size, 
                                          (self.num_targets, 2)).astype(np.float32)
        
        # 重置历史记录
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 重置GAT缓存
        self.gat_cache['features'] = None
        self.gat_cache['last_update_step'] = -1

        # 重置速度限制历史
        self.connectivity_history = []
        self.speed_limits_history = []

        # 制定episode计划
        self._plan_episode()
        
        # 返回原有格式的观察
        obs_list = self._get_obs()
        return {f"agent_{i}": obs_list[i] for i in range(self.num_agents)}, {}
    
    def _plan_episode(self):
        """制定episode拓扑变化计划"""
        if self.experiment_type != 'probabilistic':
            self.episode_plan = {'type': 'normal', 'trigger_step': None, 'executed': False}
            return
        
        # 根据概率决定episode类型
        rand = np.random.random()
        
        if rand < self.normal_probability:
            episode_type = 'normal'
            trigger_step = None
        elif rand < self.normal_probability + self.loss_probability:
            episode_type = 'loss'
            trigger_step = int(self.max_steps * (0.3 + 0.4 * np.random.random()))
        else:
            episode_type = 'addition'
            trigger_step = int(self.max_steps * (0.3 + 0.4 * np.random.random()))
        
        self.episode_plan = {
            'type': episode_type,
            'trigger_step': trigger_step,
            'executed': False
        }
        
        print(f"📋 Episode计划: {episode_type}" + 
              (f" (第{trigger_step}步触发)" if trigger_step else ""))
    
    def step(self, actions):
        """执行一步 - 添加基于连接性的速度限制"""
        self.curr_step += 1

        # 计算当前步的动态速度限制
        speed_limits = self._compute_connectivity_based_speed_limits()
        self.speed_limits_history.append(speed_limits.copy())

        # 记录速度违规情况
        speed_violations = {}

        # 执行动作（应用速度限制）
        for i, agent in enumerate(self.agents):
            if i in self.active_agents and agent in actions:
                action = np.array(actions[agent], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)

                # 计算期望速度
                desired_velocity = action * self.max_speed
                desired_speed = np.linalg.norm(desired_velocity)

                # 应用动态速度限制
                speed_limit = speed_limits[i]
                if desired_speed > speed_limit:
                    # 速度超限，按比例缩放
                    if desired_speed > 0:
                        scale_factor = speed_limit / desired_speed
                        actual_velocity = desired_velocity * scale_factor
                    else:
                        actual_velocity = desired_velocity

                    # 记录违规
                    speed_violations[agent] = {
                        'desired_speed': desired_speed,
                        'limit': speed_limit,
                        'violation': desired_speed - speed_limit
                    }
                else:
                    actual_velocity = desired_velocity
                    speed_violations[agent] = {
                        'desired_speed': desired_speed,
                        'limit': speed_limit,
                        'violation': 0.0
                    }

                # 更新速度和位置
                self.agent_vel[i] = actual_velocity
                self.agent_pos[i] += self.agent_vel[i]

                # 边界处理
                self.agent_pos[i] = np.clip(self.agent_pos[i],
                                          -self.world_size, self.world_size)

                # 记录动作
                self.prev_actions[i] = action
        
        # 检查拓扑变化
        self._check_topology_change()

        # 计算奖励（包含速度限制奖励）
        rewards = self._compute_rewards(speed_violations)
        
        # 检查结束条件
        dones = {agent: self.curr_step >= self.max_steps for agent in self.agents}
        truncated = dones.copy()
        
        # 获取观察
        obs_list = self._get_obs()
        observations = {f"agent_{i}": obs_list[i] for i in range(self.num_agents)}
        
        return observations, rewards, dones, truncated, {}
    
    def _check_topology_change(self):
        """检查并执行拓扑变化"""
        if (self.episode_plan['type'] != 'normal' and 
            not self.episode_plan['executed'] and 
            self.curr_step >= self.episode_plan['trigger_step']):
            
            if self.episode_plan['type'] == 'loss':
                success = self._execute_uav_loss()
            elif self.episode_plan['type'] == 'addition':
                success = self._execute_uav_addition()
            else:
                success = False
            
            if success:
                self.episode_plan['executed'] = True
                print(f"🎯 执行计划: {self.episode_plan['type']} (第{self.curr_step}步)")
    
    def _execute_uav_loss(self):
        """执行UAV损失"""
        if len(self.active_agents) <= self.min_active_agents:
            return False
        
        lost_uav = np.random.choice(self.active_agents)
        self.active_agents.remove(lost_uav)
        
        print(f"UAV {lost_uav} 已失效，当前位置: {self.agent_pos[lost_uav]}")
        print(f"[实验模式: UAV损失] Step {self.curr_step}: UAV {lost_uav} 失效")
        return True
    
    def _execute_uav_addition(self):
        """执行UAV添加"""
        if len(self.active_agents) >= self.max_active_agents:
            return False
        
        inactive_uavs = [i for i in range(self.num_agents) if i not in self.active_agents]
        if not inactive_uavs:
            return False
        
        new_uav = np.random.choice(inactive_uavs)
        self.active_agents.append(new_uav)
        
        # 重新初始化位置
        self.agent_pos[new_uav] = np.random.uniform(-self.world_size, self.world_size, 2)
        self.agent_vel[new_uav] = np.zeros(2)
        
        print(f"[实验模式: UAV添加] Step {self.curr_step}: 添加新UAV {new_uav}")
        return True

    def _get_obs(self):
        """获取观察 - 优化版本，减少重复计算"""
        # 计算GAT特征（使用缓存）
        gat_features = self._compute_gat_features()

        # 预计算公共数据，避免重复计算
        active_agents_ratio = len(self.active_agents) / self.num_agents
        time_progress = self.curr_step / self.max_steps

        # 批量计算目标相对位置
        target_relative_all = []
        for i in range(self.num_agents):
            target_relative = (self.target_pos - self.agent_pos[i]).flatten()
            target_relative_all.append(target_relative)

        # 批量计算邻居相对位置
        neighbor_relative_all = []
        for i in range(self.num_agents):
            neighbor_relative = []
            for j in range(self.num_agents):
                if j != i:
                    if j in self.active_agents:
                        relative_pos = self.agent_pos[j] - self.agent_pos[i]
                        neighbor_relative.extend(relative_pos)
                    else:
                        neighbor_relative.extend([0.0, 0.0])
            neighbor_relative_all.append(neighbor_relative)

        # 批量转换GAT特征
        gat_features_np = []
        for i in range(self.num_agents):
            if i < len(gat_features):
                gat_feat = gat_features[i]
                if torch.is_tensor(gat_feat):
                    if gat_feat.requires_grad:
                        gat_feat = gat_feat.detach().cpu().numpy()
                    else:
                        gat_feat = gat_feat.cpu().numpy()
                gat_features_np.append(gat_feat)
            else:
                gat_features_np.append(np.zeros(32, dtype=np.float32))

        # 组装观察
        obs_list = []
        for i in range(self.num_agents):
            obs_parts = np.concatenate([
                self.agent_pos[i],                    # 位置
                self.agent_vel[i],                    # 速度
                target_relative_all[i],               # 目标相对位置
                neighbor_relative_all[i],             # 邻居信息
                gat_features_np[i],                   # GAT特征
                [active_agents_ratio,                 # 活跃比例
                 float(i in self.active_agents),      # 自身状态
                 time_progress]                       # 时间进度
            ]).astype(np.float32)

            obs_list.append(obs_parts)

        return obs_list

    def _compute_gat_features(self):
        """计算GAT特征 - 优化版本，减少CPU-GPU传输"""
        # 检查是否需要更新GAT特征（缓存机制）
        if (self.gat_cache['features'] is not None and
            self.curr_step - self.gat_cache['last_update_step'] < self.gat_cache['update_interval']):
            return self.gat_cache['features']

        # 准备输入数据 - 批量转换减少传输次数
        uav_features = np.concatenate([self.agent_pos, self.agent_vel], axis=1)

        # 一次性传输所有数据到GPU
        with torch.no_grad():
            uav_tensor = torch.FloatTensor(uav_features).to(self.device, non_blocking=True)
            target_tensor = torch.FloatTensor(self.target_pos).to(self.device, non_blocking=True)
            uav_pos_tensor = torch.FloatTensor(self.agent_pos).to(self.device, non_blocking=True)
            target_pos_tensor = torch.FloatTensor(self.target_pos).to(self.device, non_blocking=True)

        # 创建邻接矩阵
        uav_adj, uav_target_adj = create_adjacency_matrices(
            uav_pos_tensor, target_pos_tensor,
            self.communication_range, self.sensing_range,
            active_uavs=self.active_agents
        )

        # GAT前向传播
        with torch.set_grad_enabled(self.training):
            gat_features = self.gat_model(uav_tensor, target_tensor,
                                        uav_adj, uav_target_adj,
                                        active_agents=self.active_agents)

        if not self.training:
            gat_features = gat_features.detach()

        # 更新缓存
        self.gat_cache['features'] = gat_features
        self.gat_cache['last_update_step'] = self.curr_step

        return gat_features

    def _compute_connectivity_based_speed_limits(self):
        """
        基于连接性计算每个UAV的动态速度限制
        实现论文引理1的连接性保持约束
        """
        speed_limits = np.full(self.num_agents, self.max_base_speed, dtype=np.float32)

        # 计算当前连接性矩阵
        connectivity_matrix = self._compute_connectivity_matrix()

        for i in range(self.num_agents):
            if i not in self.active_agents:
                speed_limits[i] = 0.0
                continue

            # 找到关键邻居集合 N_i^c (critical neighbors)
            critical_neighbors = self._find_critical_neighbors(i, connectivity_matrix)

            if len(critical_neighbors) == 0:
                # 没有关键邻居，使用基础速度限制
                speed_limits[i] = self.max_base_speed
            else:
                # 计算基于邻居距离的速度约束
                min_constraint = float('inf')

                for j in critical_neighbors:
                    if j in self.active_agents:
                        # 计算到邻居j的距离
                        dist_ij = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])

                        # 基于引理1的约束计算
                        # ε_i = min(r_c - d_ij) ≤ ε
                        epsilon_i = min(self.communication_range - dist_ij, self.epsilon)

                        if epsilon_i > 0:
                            # Δd_i ≤ ε_i/2 (当N_i^c ≠ ∅时)
                            constraint = epsilon_i / 2.0
                            min_constraint = min(min_constraint, constraint)

                if min_constraint != float('inf'):
                    speed_limits[i] = max(min_constraint, self.min_speed_limit)
                else:
                    speed_limits[i] = self.max_base_speed

        return speed_limits

    def _compute_connectivity_matrix(self):
        """计算连接性矩阵"""
        n = self.num_agents
        connectivity_matrix = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i+1, n):
                if i in self.active_agents and j in self.active_agents:
                    dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                    if dist <= self.communication_range:
                        connectivity_matrix[i, j] = True
                        connectivity_matrix[j, i] = True

        return connectivity_matrix

    def _find_critical_neighbors(self, agent_id, connectivity_matrix):
        """
        找到agent的关键邻居集合
        关键邻居是那些对保持全局连通性至关重要的邻居
        """
        critical_neighbors = []

        # 获取当前邻居
        current_neighbors = np.where(connectivity_matrix[agent_id])[0]

        for neighbor in current_neighbors:
            if neighbor in self.active_agents:
                # 检查移除这个连接是否会影响全局连通性
                temp_matrix = connectivity_matrix.copy()
                temp_matrix[agent_id, neighbor] = False
                temp_matrix[neighbor, agent_id] = False

                if not self._is_graph_connected(temp_matrix):
                    critical_neighbors.append(neighbor)

        return critical_neighbors

    def _is_graph_connected(self, adjacency_matrix):
        """检查图是否连通（使用DFS）"""
        active_indices = [i for i in range(self.num_agents) if i in self.active_agents]

        if len(active_indices) <= 1:
            return True

        # 从第一个活跃节点开始DFS
        visited = set()
        stack = [active_indices[0]]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # 添加所有连接的邻居
            for neighbor in range(self.num_agents):
                if (neighbor in self.active_agents and
                    neighbor not in visited and
                    adjacency_matrix[node, neighbor]):
                    stack.append(neighbor)

        return len(visited) == len(active_indices)

    def _compute_rewards(self, speed_violations=None):
        """计算奖励 - 包含速度限制相关奖励"""
        rewards = {}

        # 计算覆盖率
        coverage_rate, _, _, _ = self.calculate_coverage_complete()

        # 计算连通性
        connectivity_reward = self._compute_connectivity_reward()

        # 计算稳定性
        stability_reward = self._compute_stability_reward()

        # 计算速度合规奖励
        speed_compliance_reward = self._compute_speed_compliance_reward(speed_violations)

        # 基础奖励（加入速度合规）
        base_reward = (
            self.coverage_weight * coverage_rate +
            self.connectivity_weight * connectivity_reward +
            self.stability_weight * stability_reward +
            0.1 * speed_compliance_reward  # 速度合规权重
        )

        for i, agent in enumerate(self.agents):
            if i in self.active_agents:
                reward = base_reward

                # 边界惩罚
                if np.any(np.abs(self.agent_pos[i]) > self.world_size - 0.1):
                    reward -= self.boundary_weight

                # 个体速度违规惩罚
                if speed_violations and agent in speed_violations:
                    violation = speed_violations[agent]['violation']
                    if violation > 0:
                        reward += self.speed_violation_penalty * violation

                rewards[agent] = reward
            else:
                rewards[agent] = 0.0

        return rewards

    def _compute_speed_compliance_reward(self, speed_violations):
        """计算速度合规奖励"""
        if not speed_violations:
            return 0.0

        total_compliance = 0.0
        active_count = 0

        for agent in self.agents:
            if agent in speed_violations:
                violation = speed_violations[agent]['violation']
                # 合规度 = 1 - (违规程度 / 最大可能违规)
                compliance = max(0.0, 1.0 - violation / self.max_base_speed)
                total_compliance += compliance
                active_count += 1

        return total_compliance / max(active_count, 1)

    def calculate_coverage_complete(self):
        """计算完整覆盖率信息 - 保持原有接口"""
        covered_targets = 0
        for target_pos in self.target_pos:
            for agent_idx in self.active_agents:
                distance = np.linalg.norm(target_pos - self.agent_pos[agent_idx])
                if distance <= self.sensing_range:
                    covered_targets += 1
                    break

        coverage_rate = covered_targets / self.num_targets if self.num_targets > 0 else 0.0
        is_fully_connected = True  # 简化实现
        max_coverage_rate = coverage_rate
        unconnected_uav = 0

        return coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav

    def _compute_connectivity_reward(self):
        """计算连通性奖励"""
        if len(self.active_agents) <= 1:
            return 1.0

        connected_count = 0
        for i in self.active_agents:
            for j in self.active_agents:
                if i != j:
                    distance = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                    if distance <= self.communication_range:
                        connected_count += 1
                        break

        return connected_count / len(self.active_agents)

    def _compute_stability_reward(self):
        """计算稳定性奖励"""
        # 简化的稳定性计算
        if len(self.coverage_history) < 2:
            return 0.0

        recent_coverage = self.coverage_history[-10:] if len(self.coverage_history) >= 10 else self.coverage_history
        if len(recent_coverage) < 2:
            return 0.0

        stability = 1.0 - np.std(recent_coverage)
        return max(0.0, stability)

    def render(self, mode='human'):
        """渲染环境 - 保持原有接口"""
        if mode == 'rgb_array':
            return self._render_rgb_array()
        elif mode == 'human':
            return self._render_human()
        else:
            return None

    def _render_human(self):
        """人类可视化渲染"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        # 清屏
        self.screen.fill((255, 255, 255))

        # 绘制目标
        for target_pos in self.target_pos:
            screen_pos = self._world_to_screen(target_pos)
            pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, 8)

        # 绘制UAV
        for i in self.active_agents:
            screen_pos = self._world_to_screen(self.agent_pos[i])
            color = (0, 0, 255) if i in self.active_agents else (128, 128, 128)
            pygame.draw.circle(self.screen, color, screen_pos, 12)

            # 绘制感知范围
            sensing_radius = int(self.sensing_range * self.screen_size / (2 * self.world_size))
            pygame.draw.circle(self.screen, (0, 255, 0), screen_pos, sensing_radius, 1)

        # 显示实验信息
        font = pygame.font.Font(None, 24)
        experiment_text = f"实验类型: {self.experiment_type}"
        text_surface = font.render(experiment_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        step_text = f"步数: {self.curr_step}/{self.max_steps}"
        text_surface = font.render(step_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 35))

        uav_text = f"活跃UAV: {len(self.active_agents)}/{self.num_agents}"
        text_surface = font.render(uav_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 60))

        pygame.display.flip()
        self.clock.tick(60)

    def _render_rgb_array(self):
        """返回RGB数组"""
        if self.screen is None:
            self._render_human()

        # 获取屏幕内容
        rgb_array = pygame.surfarray.array3d(self.screen)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        return rgb_array

    def _world_to_screen(self, world_pos):
        """世界坐标转屏幕坐标"""
        x = int((world_pos[0] + self.world_size) / (2 * self.world_size) * self.screen_size)
        y = int((world_pos[1] + self.world_size) / (2 * self.world_size) * self.screen_size)
        return (x, y)

    # GAT相关方法 - 保持原有接口
    def get_gat_parameters(self):
        """获取GAT参数"""
        return self.gat_model.parameters()

    def save_gat_model(self, path):
        """保存GAT模型"""
        torch.save(self.gat_model.state_dict(), path)

    def load_gat_model(self, path):
        """加载GAT模型"""
        self.gat_model.load_state_dict(torch.load(path, map_location=self.device))

    # 兼容性方法 - 保持原有接口
    def get_observation_space(self, agent):
        """获取观察空间"""
        return self.observation_spaces[agent]

    def get_action_space(self, agent):
        """获取动作空间"""
        return self.action_spaces[agent]

    def fail_uav(self, uav_idx):
        """使UAV失效 - 兼容性方法"""
        if uav_idx in self.active_agents:
            self.active_agents.remove(uav_idx)
            return True
        return False

    def add_uav(self):
        """添加UAV - 兼容性方法"""
        inactive_uavs = [i for i in range(self.num_agents) if i not in self.active_agents]
        if inactive_uavs:
            new_uav = inactive_uavs[0]
            self.active_agents.append(new_uav)
            self.agent_pos[new_uav] = np.random.uniform(-self.world_size, self.world_size, 2)
            self.agent_vel[new_uav] = np.zeros(2)
            return new_uav
        return None

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
