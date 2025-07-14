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
import networkx as nx

# 动态导入GAT模型
import importlib.util
import os

def _import_gat_model():
    """动态导入GAT模型"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 修改为导入带LSTM的GAT模型
    gat_model_path = os.path.join(current_dir, 'gat_model_top_lstm.py')

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
        self.curr_step = 0
        self.max_coverage_rate = 0.0
        
        # 物理参数
        self.max_speed = 3.0
        self.communication_range = 1.0
        self.coverage_radius = 0.5
        self.dt = 0.1
        
        # 拓扑参数
        self.min_active_agents = min_active_agents
        self.max_active_agents = max_active_agents or num_agents
        
        # 新的奖励权重 (优化连通性和覆盖平衡)
        self.connectivity_weight = 2.0      # 连通性权重 (提高)
        self.coverage_weight = 15.0         # 覆盖权重 (适当降低)
        self.topology_weight = 5.0          # 拓扑适应权重
        self.boundary_weight = 1.0          # 边界权重
        self.critical_connection_weight = 0.5  # 关键连接权重 (新增)
        
        # 拓扑实验概率
        self.normal_probability = 0.60
        self.loss_probability = 0.25
        self.addition_probability = 0.15
        
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

        # GAT缓存
        self.gat_cache = {
            'features': None,
            'last_update_step': -1,
            'update_interval': 3
        }

        # 速度限制参数
        self.max_base_speed = 1.0
        self.connectivity_speed_factor = 0.5
        self.min_speed_limit = 0.2
        self.epsilon = 0.1
        self.speed_violation_penalty = -5.0

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

        # 设置GAT模型为训练模式
        self.gat_model.train()
        
        # 训练模式标志
        self.training = True  # 设置为True，让GAT参与训练
        
        # 观察和动作空间（保持原有格式）
        self._setup_spaces()
        
        # 渲染相关 - 提高分辨率以改善视频画质
        self.screen = None
        self.clock = None
        self.width, self.height = (800, 800)  # 从700x700提升到1200x1200
        self.metadata = {"render_fps": 60}
        self.font = None
        
        # 历史记录
        self.coverage_history = []
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 加速度控制参数
        self.max_acceleration = 0.4  # 最大加速度
        self.damping_factor = 0.95   # 阻尼系数（可选）

        # 训练步数计数器
        self.training_step = 0

        # 覆盖奖励参数
        self.covered_targets = set()  # 已覆盖的目标集合
        self.unique_coverage_weight = 15.0  # 唯一覆盖权重
        
        # 优先目标点参数
        self.priority_targets_ratio = 0.5 # 例如，20%的目标是优先目标
        self.num_priority_targets = int(self.num_targets * self.priority_targets_ratio)
        self.priority_target_indices = []  # 将在reset中初始化
        self.priority_coverage_weight = 25.0  # 优先目标点覆盖权重
        
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
        self.max_coverage_rate = 0.0
        self.active_agents = list(range(self.num_agents))

        # 重置奖励相关参数
        self.covered_targets = set()  # 重置已覆盖目标集合
        
        # 初始化UAV位置（与原版本一致：底部排列）
        self.agent_pos = []
        bottom_y = -self.world_size + 0.15
        spacing = 2 * self.world_size / (self.num_agents + 1)
        for i in range(self.num_agents):
            x = -self.world_size + (i + 1) * spacing
            self.agent_pos.append([x, bottom_y])
        self.agent_pos = np.array(self.agent_pos, dtype=np.float32)

        self.agent_vel = np.zeros((self.num_agents, 2), dtype=np.float32)

        # 初始化目标位置（与原版本一致：0.85范围内随机分布）
        self.target_pos = np.random.uniform(-self.world_size*0.85, self.world_size*0.85,
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
        
        # 重置LSTM状态
        if hasattr(self.gat_model, 'reset_lstm_states'):
            self.gat_model.reset_lstm_states()
        
        # 随机选择优先目标点
        self.priority_target_indices = np.random.choice(
            self.num_targets, 
            size=self.num_priority_targets, 
            replace=False
        ).tolist()
        
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
        """执行一步 - 使用加速度控制"""
        self.curr_step += 1

        # 计算当前步的动态速度限制
        speed_limits = self._compute_connectivity_based_speed_limits()
        self.speed_limits_history.append(speed_limits.copy())

        # 记录速度违规情况
        speed_violations = {}

        # 执行动作（加速度控制）
        for i, agent in enumerate(self.agents):
            if i in self.active_agents and agent in actions:
                action = np.array(actions[agent], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)

                # 更新智能体速度和位置
                self._update_agent_dynamics(i, action, speed_limits[i], speed_violations, agent)

                # 记录原始动作
                self.prev_actions[i] = action
        
        # 检查拓扑变化
        self._check_topology_change()

        # 计算奖励
        rewards = self._compute_rewards()
        
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
            self.communication_range, self.coverage_radius,
            active_uavs=self.active_agents
        )

        # 为LSTM准备智能体ID和重置状态标志
        agent_ids = list(range(self.num_agents))
        reset_states = (self.curr_step == 0)  # 第一步重置状态

        # GAT+LSTM前向传播
        with torch.set_grad_enabled(self.training):
            gat_features = self.gat_model(uav_tensor, target_tensor,
                                        uav_adj, uav_target_adj,
                                        active_agents=self.active_agents,
                                        agent_ids=agent_ids,
                                        reset_states=reset_states)

        if not self.training:
            gat_features = gat_features.detach()

        # 更新缓存
        self.gat_cache['features'] = gat_features
        self.gat_cache['last_update_step'] = self.curr_step

        return gat_features

    def _compute_connectivity_based_speed_limits(self):
        """
        简化的连接性速度限制 - 优化强化学习训练
        只在关键情况下进行温和限制
        """
        speed_limits = np.full(self.num_agents, self.max_base_speed, dtype=np.float32)

        # 计算当前连接性矩阵
        connectivity_matrix = self._compute_connectivity_matrix()

        for i in range(self.num_agents):
            if i not in self.active_agents:
                speed_limits[i] = 0.0
                continue

            # 找到关键邻居集合
            critical_neighbors = self._find_critical_neighbors(i, connectivity_matrix)

            # 简化的速度限制策略
            if len(critical_neighbors) == 0:
                # 没有关键邻居，无速度限制
                speed_limits[i] = self.max_base_speed
            elif len(critical_neighbors) == 1:
                # 只有1个关键邻居，轻微限制
                speed_limits[i] = self.max_base_speed * 0.8  # 90%速度
            elif len(critical_neighbors) == 2:
                # 2个关键邻居，中等限制
                speed_limits[i] = self.max_base_speed * 0.7  # 80%速度
            else:
                # 3个或更多关键邻居，较强限制（避免成为关键节点）
                speed_limits[i] = self.max_base_speed * 0.5  # 60%速度

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

    def _compute_rewards(self):
        """计算奖励"""
        self.training_step += 1
        rewards = {}

        # 1. 计算全局奖励组件
        connectivity_reward = self._compute_connectivity_reward_advanced()
        coverage_reward = self._compute_coverage_reward_advanced()
        topology_reward = self._compute_topology_adaptation_reward()
        critical_connection_reward = self._compute_critical_connection_reward()  # 新增关键连接奖励
        priority_target_reward = self._compute_priority_target_reward() # 新增优先目标点奖励

        # 2. 全局奖励
        global_reward = (connectivity_reward * self.connectivity_weight +
                        coverage_reward * self.coverage_weight +
                        topology_reward * self.topology_weight +
                        critical_connection_reward * self.critical_connection_weight +
                        priority_target_reward * self.priority_coverage_weight) # 添加优先目标点奖励权重

        # 3. 为每个智能体分配奖励
        for i, agent in enumerate(self.agents):
            if i in self.active_agents:
                # 个体奖励 = 全局奖励 + 个体特定奖励
                individual_reward = self._compute_individual_reward(i)
                boundary_penalty = self._calculate_boundary_penalty(i)

                total_reward = global_reward + individual_reward + boundary_penalty

                # 时间惩罚
                time_penalty = -0.05 * self.curr_step
                total_reward += time_penalty
                
                rewards[agent] = total_reward / 100.0  # 缩放到合理范围
            else:
                rewards[agent] = 0.0

        return rewards

    def _compute_connectivity_reward_advanced(self):
        """基于图论的连通性奖励 - 简化版"""
        # 构建通信图
        G = nx.Graph()
        active_agents = list(self.active_agents)

        # 添加节点和边
        for i in active_agents:
            G.add_node(i)

        for i in active_agents:
            for j in active_agents:
                if i >= j:
                    continue
                dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                if dist <= self.communication_range:
                    G.add_edge(i, j)

        if len(active_agents) <= 1:
            return 0

        # 1. 基础连通性奖励/惩罚
        if nx.is_connected(G):
            connectivity_reward = 3.0  # 连通奖励
        else:
            connectivity_reward = -2.0  # 不连通严厉惩罚

        # 2. 图结构质量奖励 (避免过度连接)
        structure_reward = 0
        if nx.is_connected(G):
            n_nodes = len(active_agents)
            min_edges = n_nodes - 1  # 最小生成树边数
            actual_edges = G.number_of_edges()

            if actual_edges == min_edges:
                structure_reward = 2.0  # 最优结构 (树状)
            elif actual_edges <= min_edges * 1.5:
                structure_reward = 1.0  # 适度冗余
            else:
                structure_reward = -1.0  # 过度连接惩罚

        return connectivity_reward + structure_reward

    def _compute_coverage_reward_advanced(self):
        """计算覆盖奖励 - 移除距离矛盾"""
        # 1. 计算覆盖率
        coverage_rate, _, _, _ = self.calculate_coverage_complete()

        # 2. 如果没有活跃UAV，返回0
        if len(self.active_agents) == 0:
            return 0

        # 3. 简化奖励 - 仅使用覆盖率的平方
        r_s = coverage_rate ** 2.0

        # 4. 权重调整 - 保留原有的智能体数量缩放
        k_1 = 35 * len(self.active_agents)
        return k_1 * r_s



    def _compute_topology_adaptation_reward(self):
        """计算拓扑适应奖励"""
        if not hasattr(self, 'episode_plan'):
            return 0

        # 如果发生了拓扑变化
        if self.episode_plan.get('executed', False):
            coverage_rate, _, _, _ = self.calculate_coverage_complete()

            # 拓扑变化后的适应性奖励
            if coverage_rate > 0.7:  # 快速恢复高覆盖率
                return 10.0
            elif coverage_rate > 0.5:  # 中等恢复
                return 5.0
            else:  # 恢复较慢
                return -2.0

        return 0

    def _compute_individual_reward(self, agent_idx):
        """计算个体奖励 - 专注覆盖，移除距离奖励避免聚集"""
        reward = 0

        # 1. 唯一覆盖奖励
        unique_coverage_reward = 0.0
        agent_pos = self.agent_pos[agent_idx]

        for target_pos in self.target_pos:
            if np.linalg.norm(agent_pos - target_pos) < self.coverage_radius:
                # 检查是否只有当前无人机覆盖该目标点
                covered_by_others = False
                for other_idx in self.active_agents:
                    if other_idx == agent_idx:
                        continue
                    if np.linalg.norm(self.agent_pos[other_idx] - target_pos) < self.coverage_radius:
                        covered_by_others = True
                        break

                if not covered_by_others:
                    unique_coverage_reward += 1.0

        reward += unique_coverage_reward * self.unique_coverage_weight

        """
        移除距离奖励！让连通性由全局奖励处理，避免UAV聚集倾向
        这样UAV可以自由分散探索，不会为了获得连接奖励而聚集
        """
        return reward

    def _compute_critical_connection_reward(self):
        """关键连接奖励 - 奖励维持关键连接的UAV，避免过度依赖"""
        reward = 0
        connectivity_matrix = self._compute_connectivity_matrix()

        for i in self.active_agents:
            critical_neighbors = self._find_critical_neighbors(i, connectivity_matrix)

            # 奖励维持关键连接的UAV
            if len(critical_neighbors) > 0:
                reward += len(critical_neighbors) * 1.0  # 每个关键连接+1

            # 惩罚成为单点故障的UAV (避免过度依赖单个UAV)
            if len(critical_neighbors) > 2:
                reward -= 0.5  # 避免网络过度依赖单个节点

        return reward

    def _compute_priority_target_reward(self):
        """计算优先目标点覆盖奖励"""
        if not self.priority_target_indices or len(self.active_agents) == 0:
            return 0.0
            
        # 计算优先目标点的覆盖情况
        priority_covered_count = 0
        for i in self.priority_target_indices:
            target_pos = self.target_pos[i]
            for agent_idx in self.active_agents:
                agent_pos = self.agent_pos[agent_idx]
                if np.linalg.norm(target_pos - agent_pos) <= self.coverage_radius:
                    priority_covered_count += 1
                    break
        
        # 计算优先目标点覆盖率
        priority_coverage_rate = priority_covered_count / len(self.priority_target_indices)
        
        # 使用平方奖励，鼓励更高的覆盖率
        reward = (priority_coverage_rate ** 2) * 10.0
        
        # 如果全部覆盖，给予额外奖励
        if priority_coverage_rate >= 1.0:
            reward += 5.0
            
        return reward

    def _calculate_boundary_penalty(self, agent_idx):
        """
        计算边界惩罚 - 移植并适配自动态避障实验
        适配您的拓扑UAV环境，简化并调整参数
        """
        penalty = 0.0
        agent_pos = self.agent_pos[agent_idx]

        # 适配参数 (比原版温和)
        max_penalty = 50.0           # 降低惩罚上限
        boundary_limit = self.world_size  # 使用环境的world_size (1.0)
        safe_margin = 0.15            # 安全边距
        penalty_factor = 20.0        # 降低惩罚因子
        penalty_exponent = 2.5       # 降低指数斜率

        # 检查每个维度 (x, y)
        for dim in range(2):
            pos = agent_pos[dim]

            # 1. 检查接近负边界 (pos < -boundary_limit + safe_margin)
            if pos < -boundary_limit + safe_margin:
                distance_to_boundary = (-boundary_limit + safe_margin) - pos
                if distance_to_boundary > 0:
                    # 温和的指数惩罚
                    penalty -= penalty_factor * np.exp(penalty_exponent * distance_to_boundary)
                    penalty = np.clip(penalty, -max_penalty, 0)

            # 2. 检查超出负边界 (pos < -boundary_limit)
            if pos < -boundary_limit:
                distance_outside = -boundary_limit - pos
                # 严厉的超出边界惩罚
                penalty -= penalty_factor * 2 * np.exp(penalty_exponent * distance_outside)
                penalty = np.clip(penalty, -max_penalty * 2, 0)

            # 3. 检查接近正边界 (pos > boundary_limit - safe_margin)
            if pos > boundary_limit - safe_margin:
                distance_to_boundary = pos - (boundary_limit - safe_margin)
                if distance_to_boundary > 0:
                    # 温和的指数惩罚
                    penalty -= penalty_factor * np.exp(penalty_exponent * distance_to_boundary)
                    penalty = np.clip(penalty, -max_penalty, 0)

            # 4. 检查超出正边界 (pos > boundary_limit)
            if pos > boundary_limit:
                distance_outside = pos - boundary_limit
                # 严厉的超出边界惩罚
                penalty -= penalty_factor * 2 * np.exp(penalty_exponent * distance_outside)
                penalty = np.clip(penalty, -max_penalty * 2, 0)

        return penalty



    def _update_agent_dynamics(self, agent_idx, action, speed_limit, speed_violations, agent_name):
        """更新单个智能体的动力学状态（加速度控制）"""
        # 1. 动作转换为加速度
        acceleration = action * self.max_acceleration

        # 2. 更新速度（积分）
        new_velocity = self.agent_vel[agent_idx] + acceleration * self.dt

        # 3. 应用阻尼
        new_velocity *= self.damping_factor

        # 4. 应用连接性速度限制
        current_speed = np.linalg.norm(new_velocity)

        if current_speed > speed_limit:
            # 速度超限，按比例缩放
            if current_speed > 0:
                scale_factor = speed_limit / current_speed
                actual_velocity = new_velocity * scale_factor
            else:
                actual_velocity = new_velocity

            # 记录违规
            speed_violations[agent_name] = {
                'desired_speed': current_speed,
                'limit': speed_limit,
                'violation': current_speed - speed_limit
            }
        else:
            actual_velocity = new_velocity
            speed_violations[agent_name] = {
                'desired_speed': current_speed,
                'limit': speed_limit,
                'violation': 0.0
            }

        # 5. 应用最大速度限制
        final_speed = np.linalg.norm(actual_velocity)
        if final_speed > self.max_speed:
            actual_velocity = actual_velocity * (self.max_speed / final_speed)

        # 6. 更新速度和位置
        self.agent_vel[agent_idx] = actual_velocity
        self.agent_pos[agent_idx] += self.agent_vel[agent_idx] * self.dt

        # 7. 边界处理
        self.agent_pos[agent_idx] = np.clip(self.agent_pos[agent_idx],
                                          -self.world_size, self.world_size)

    def close(self):
        """关闭环境 - 与原版本一致"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def calculate_coverage_complete(self):
        """计算完整覆盖率信息 - 修复：只计算活跃UAV的覆盖"""
        # 计算目标点是否被覆盖 (只考虑活跃UAV)
        covered_flags = []
        for target in self.target_pos:
            covered = False
            for i in self.active_agents:  # 修复：只检查活跃UAV
                agent_pos = self.agent_pos[i]
                distance = np.linalg.norm(target - agent_pos)
                if distance <= self.coverage_radius:
                    covered = True
                    break
            covered_flags.append(covered)

        # 计算覆盖率
        covered_count = sum(covered_flags)
        total_targets = len(self.target_pos)
        coverage_rate = covered_count / total_targets if total_targets > 0 else 0

        # 构建通信邻接矩阵 (只考虑活跃UAV)
        active_agents = list(self.active_agents)
        num_active = len(active_agents)

        if num_active <= 1:
            # 0或1个活跃UAV，连通性定义为True
            fully_connected = True
            unconnected_count = 0
        else:
            # 构建活跃UAV之间的邻接矩阵
            adjacency_matrix = np.zeros((num_active, num_active), dtype=bool)
            for i in range(num_active):
                for j in range(i + 1, num_active):
                    agent_i = active_agents[i]
                    agent_j = active_agents[j]
                    distance = np.linalg.norm(self.agent_pos[agent_i] - self.agent_pos[agent_j])
                    if distance <= self.communication_range:
                        adjacency_matrix[i, j] = True
                        adjacency_matrix[j, i] = True

            # DFS 检查连通性 (从第一个活跃UAV开始)
            visited = [False] * num_active

            def dfs(idx):
                visited[idx] = True
                for neighbor_idx, connected in enumerate(adjacency_matrix[idx]):
                    if connected and not visited[neighbor_idx]:
                        dfs(neighbor_idx)

            dfs(0)  # 从第一个活跃UAV开始
            fully_connected = all(visited)
            unconnected_count = visited.count(False)

        # 如果是优先目标点，记录到优先覆盖列表中
        priority_covered_flags = []
        for i, tpos in enumerate(self.target_pos):
            covered = False
            for agent_idx in self.active_agents:
                agent_pos = self.agent_pos[agent_idx]
                if np.linalg.norm(tpos - agent_pos) <= self.coverage_radius:
                    covered = True
                    break
            if i in self.priority_target_indices:
                priority_covered_flags.append(covered)

        # 计算优先目标点覆盖率
        priority_covered_count = sum(priority_covered_flags)
        total_priority_targets = len(self.priority_target_indices)
        priority_coverage_rate = priority_covered_count / total_priority_targets if total_priority_targets > 0 else 0

        # 更新最大覆盖率
        if fully_connected:
            self.max_coverage_rate = max(self.max_coverage_rate, coverage_rate)

        return coverage_rate, fully_connected, self.max_coverage_rate, unconnected_count, priority_coverage_rate



    def render(self):
        """渲染环境 - 与原版本完全一致"""
        if self.render_mode is None:
            import gym
            gym.logger.warn(
                "Calling render without specifying render_mode."
            )
            return
        if self.screen is None:
            pygame.init()
            pygame.font.init()  # 确保字体模块初始化
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('UAV Topology')
            self.clock = pygame.time.Clock()

            # 初始化字体
            try:
                # 尝试使用系统字体
                self.font = pygame.font.SysFont('arial', 24, bold=True)
                if self.font is None:
                    raise Exception("SysFont failed")
            except:
                try:
                    # 备选：使用默认字体
                    self.font = pygame.font.Font(None, 28)
                except:
                    # 最后备选：创建基础字体
                    self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.draw()
        if self.render_mode == 'rgb_array':
            data = pygame.surfarray.array3d(self.screen)
            return np.transpose(data, (1, 0, 2))
        elif self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def draw(self):
        """绘制函数 - 与原版本完全一致"""
        fixed_cam = self.world_size
        self.screen.fill((255, 255, 255))

        def to_screen(pos):
            x, y = pos
            y = -y  # y 轴翻转
            sx = int((x / fixed_cam) * (self.width / 2) + self.width / 2)
            sy = int((y / fixed_cam) * (self.height / 2) + self.height / 2)
            return sx, sy

        # 画目标点 - 区分普通目标点和优先目标点
        for i, tpos in enumerate(self.target_pos):
            sx, sy = to_screen(tpos)
            if i in self.priority_target_indices:
                # 优先目标点 - 使用橙红色
                pygame.draw.circle(self.screen, (255, 69, 0), (sx, sy), 7)
            else:
                # 普通目标点 - 使用绿色
                pygame.draw.circle(self.screen, (0, 255, 0), (sx, sy), 5)

        # 存储无人机屏幕位置用于连线
        screen_positions = []

        for i, apos in enumerate(self.agent_pos):
            sx, sy = to_screen(apos)
            screen_positions.append((sx, sy))

            # 绘制无人机图像或默认图形
            if not hasattr(self, 'drone_image'):
                path = os.path.dirname(__file__)
                img = os.path.join(path, 'UAV.png')
                if os.path.exists(img):
                    self.drone_image = pygame.transform.scale(pygame.image.load(img), (30, 30))
                else:
                    self.drone_image = None
            if self.drone_image:
                rect = self.drone_image.get_rect(center=(sx, sy))
                self.screen.blit(self.drone_image, rect)
            else:
                # 根据UAV是否失效选择颜色
                if i in self.active_agents:
                    color = (0, 0, 255)  # 蓝色 - 活跃UAV
                else:
                    color = (128, 128, 128)  # 灰色 - 失效UAV
                pygame.draw.circle(self.screen, color, (sx, sy), 8)

            # 绘制探测半径圆圈（实线）
            coverage_radius_px = int((self.coverage_radius / fixed_cam) * (self.width/2))
            pygame.draw.circle(self.screen, (0, 0, 255), (sx, sy), coverage_radius_px, 1)

            # 绘制通信半径圆圈（蓝色虚线）
            comm_radius_px = int((self.communication_range / fixed_cam) * (self.width/2))
            # 创建虚线效果
            num_segments = 80  # 虚线段数
            for j in range(num_segments):
                if j % 2 == 0:  # 只画偶数段，形成虚线
                    start_angle = 2 * np.pi * j / num_segments
                    end_angle = 2 * np.pi * (j + 1) / num_segments
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

        # 画红线：只在活跃无人机之间显示通信连接
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # 只有两个UAV都是活跃状态才显示通信线
                if i in self.active_agents and j in self.active_agents:
                    dist = np.linalg.norm(self.agent_pos[i] - self.agent_pos[j])
                    if dist <= self.communication_range:
                        pygame.draw.line(self.screen, (255, 0, 0),
                                        screen_positions[i], screen_positions[j], 1)

        # 显示文字信息（使用预初始化的字体）
        if self.font is None:
            # 如果字体未初始化，使用简单备选
            font = pygame.font.Font(None, 24)
        else:
            font = self.font

        # 显示实验类型（使用英文避免编码问题）
        experiment_text = f"Experiment: {self.experiment_type}"
        text_surface = font.render(experiment_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # 显示步数
        step_text = f"Step: {self.curr_step}/{self.max_steps}"
        text_surface = font.render(step_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 40))

        # 显示活跃UAV数量
        uav_text = f"Active UAVs: {len(self.active_agents)}/{self.num_agents}"
        text_surface = font.render(uav_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 70))

        # 显示覆盖率信息
        coverage_rate, _, _, _, priority_coverage_rate = self.calculate_coverage_complete()
        coverage_text = f"Coverage: {coverage_rate:.3f}"
        text_surface = font.render(coverage_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 100))

        # 显示优先目标点覆盖率
        priority_coverage_text = f"Priority Coverage: {priority_coverage_rate:.3f}"
        text_surface = font.render(priority_coverage_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 130))

        # 显示episode计划信息
        if hasattr(self, 'episode_plan'):
            plan_type = self.episode_plan.get('type', 'unknown')
            trigger_step = self.episode_plan.get('trigger_step', None)
            executed = self.episode_plan.get('executed', False)

            plan_text = f"Plan: {plan_type}"
            if trigger_step:
                plan_text += f" (step {trigger_step})"
            if executed:
                plan_text += " [done]"

            text_surface = font.render(plan_text, True, (0, 0, 255))
            self.screen.blit(text_surface, (10, 160))



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
            # 清零失效UAV的速度和历史动作
            if uav_idx < len(self.agent_vel):
                self.agent_vel[uav_idx] = np.zeros(2, dtype=np.float32)
            if uav_idx < len(self.prev_actions):
                self.prev_actions[uav_idx] = np.zeros(2, dtype=np.float32)
            return True
        return False

    def add_uav(self):
        """添加UAV - 兼容性方法"""
        inactive_uavs = [i for i in range(self.num_agents) if i not in self.active_agents]
        if inactive_uavs:
            new_uav = inactive_uavs[0]
            self.active_agents.append(new_uav)
            self.agent_pos[new_uav] = np.random.uniform(-self.world_size, self.world_size, 2)
            self.agent_vel[new_uav] = np.zeros(2)  # 新UAV从静止开始
            # 重置新激活UAV的历史动作
            if new_uav < len(self.prev_actions):
                self.prev_actions[new_uav] = np.zeros(2, dtype=np.float32)
            return new_uav
        return None

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def _force_episode_mode(self, mode):
        """强制设置episode模式 - 用于测试"""
        if mode == 'normal':
            self.episode_plan = {
                'type': 'normal',
                'trigger_step': None,
                'executed': False
            }
        elif mode == 'loss':
            # 设置UAV损失模式，在中期触发
            trigger_step = np.random.randint(100, 300)
            self.episode_plan = {
                'type': 'loss',
                'trigger_step': trigger_step,
                'executed': False
            }
        elif mode == 'addition':
            # 设置UAV增加模式，在中期触发
            trigger_step = np.random.randint(100, 300)
            self.episode_plan = {
                'type': 'addition',
                'trigger_step': trigger_step,
                'executed': False
            }

        print(f"🔧 强制设置模式: {mode}" +
              (f" (触发步数: {self.episode_plan['trigger_step']})" if self.episode_plan['trigger_step'] else ""))
