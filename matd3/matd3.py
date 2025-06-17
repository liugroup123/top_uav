import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import random
from torch_geometric.nn import GATConv

# 设置cudnn基准模式加速卷积（对全连接层影响有限但无害）
torch.backends.cudnn.benchmark = True

def adj_matrix_to_edge_index(adj_matrix):
    """
    将邻接矩阵转换为边索引格式
    输出格式: [2, num_edges]，表示边的源节点和目标节点
    """
    edges = torch.nonzero(adj_matrix).t().contiguous()
    # 添加自循环
    num_nodes = adj_matrix.size(0)
    self_loops = torch.arange(num_nodes, device=adj_matrix.device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edges = torch.cat([edges, self_loops], dim=1)
    return edges

# 添加GAT层实现
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha

        # 定义可训练参数
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * n_heads)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, adj):
        # x: [batch_size, N, in_features]
        # adj: [batch_size, N, N]
        batch_size = x.size(0)
        N = x.size(1)
        
        # 线性变换
        h = torch.matmul(x, self.W)  # [batch_size, N, out_features * n_heads]
        h = h.view(batch_size, N, self.n_heads, self.out_features)  # [batch_size, N, n_heads, out_features]
        
        # 准备attention计算
        a_input = torch.cat([
            h.repeat_interleave(N, dim=1),  # [batch_size, N*N, n_heads, out_features]
            h.repeat(1, N, 1, 1)  # [batch_size, N*N, n_heads, out_features]
        ], dim=-1)  # [batch_size, N*N, n_heads, 2*out_features]
        
        # 计算attention scores
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch_size, N*N, n_heads]
        e = e.view(batch_size, N, N, self.n_heads)  # [batch_size, N, N, n_heads]
        
        # Mask attention scores based on adjacency matrix
        adj = adj.unsqueeze(-1).expand_as(e)  # [batch_size, N, N, n_heads]
        e = e.masked_fill(adj == 0, float('-inf'))
        
        # 应用softmax
        attention = F.softmax(e, dim=2)  # [batch_size, N, N, n_heads]
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用attention
        h_prime = torch.einsum('bijk,bjkl->bikl', attention, h)  # [batch_size, N, n_heads, out_features]
        
        return h_prime.mean(dim=2)  # [batch_size, N, out_features]

# Actor网络保持不变
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
        super(Actor, self).__init__()
        # 使用ModuleList动态构建网络层
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        prev_dim = obs_dim
        
        # 动态构建隐藏层
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # 输出层
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # 添加残差连接（修改这里，确保维度匹配）
        self.residual_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.residual_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
    def forward(self, obs):
        x = obs
        prev_x = None
        
        # 第一层特殊处理
        x = F.leaky_relu(self.layer_norms[0](self.layers[0](x)), negative_slope=0.01)
        
        # 处理中间层，添加残差连接
        for i in range(1, len(self.layers)):
            identity = x  # 保存当前层的输入
            x = self.layers[i](x)
            x = self.layer_norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.01)
            
            # 添加残差连接（只在维度匹配的层之间）
            if i-1 < len(self.residual_layers):
                residual = self.residual_layers[i-1](identity)
                x = x + residual
        
        return torch.tanh(self.output_layer(x))

# 修改Critic网络以使用GAT
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, n_agents=3, hidden_size=64):
        super(Critic, self).__init__()
        self.n_agents = n_agents
        self.agent_obs_dim = total_obs_dim // n_agents
        self.agent_action_dim = total_action_dim // n_agents
        self.input_dim = self.agent_obs_dim + self.agent_action_dim
        
        # GAT层处理智能体间的交互
        self.gat1 = GATConv(
            in_channels=self.input_dim,
            out_channels=hidden_size,
            heads=1,  # 进一步减少注意力头
            dropout=0.05  # 减少dropout
        )
        self.gat2 = GATConv(
            in_channels=hidden_size * 2,  # 修改这里：现在第一层有2个头，所以是 hidden_size * 2
            out_channels=hidden_size,
            heads=1,
            dropout=0.1
        )
        
        # 特征处理层
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01)
        )
        
        # 输出层
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * n_agents, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def _batch_adj_to_edge_index(self, adj_matrix):
        """将批量邻接矩阵转换为边索引"""
        batch_size = adj_matrix.size(0)
        edge_indices = []
        
        for i in range(batch_size):
            edge_index = adj_matrix_to_edge_index(adj_matrix[i])
            edge_indices.append(edge_index)
            
        return torch.cat(edge_indices, dim=1)
        
    def forward(self, obs, action, adj_matrix):
        batch_size = obs.size(0)
        
        # 批量处理所有数据
        x = torch.cat([
            obs.view(batch_size, self.n_agents, -1),
            action.view(batch_size, self.n_agents, -1)
        ], dim=-1)
        
        # 一次性处理整个批次
        x = x.view(-1, self.input_dim)
        edge_index = self._batch_adj_to_edge_index(adj_matrix)
        
        # 使用批处理版本的GAT
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(self.n_agents)
        h = F.elu(self.gat1(x, edge_index, batch=batch))
        h = self.gat2(h, edge_index, batch=batch)
        
        # 重塑输出
        h = h.view(batch_size, self.n_agents, -1)
        h = h.view(batch_size, -1)  # 展平所有特征
        
        # 最终输出
        return self.output_net(h)

# 改进的Replay Buffer实现
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, alpha=0.6, beta=0.4):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0

    def add(self, obs, actions, rewards, next_obs, dones):
        """
        添加新的经验到缓冲区
        """
        experience = (obs, actions, rewards, next_obs, dones)
        # 新经验使用最大优先级
        self.tree.add(self.max_priority, experience)

    def sample(self):
        """采样一个批次的经验"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        # 更新beta值
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        # 计算重要性采样权重
        priorities = np.array(priorities)
        weights = self._compute_weights(priorities)

        return self._process_batch(batch, indices, weights)

    def _compute_weights(self, priorities):
        """计算重要性采样权重"""
        # 将优先级转换为概率
        probs = priorities / self.tree.total()
        
        # 计算重要性采样权重
        weights = (self.tree.size * probs) ** (-self.beta)
        
        # 归一化权重
        weights = weights / weights.max()
        
        return torch.as_tensor(weights, device=self.device, dtype=torch.float32)

    def _process_batch(self, batch, indices, weights):
        """处理批量数据"""
        # 解包数据
        obs_batch = defaultdict(list)
        action_batch = defaultdict(list)
        reward_batch = defaultdict(list)
        next_obs_batch = defaultdict(list)
        done_batch = defaultdict(list)

        for experience in batch:
            obs, actions, rewards, next_obs, dones = experience
            for agent in obs:
                obs_batch[agent].append(obs[agent])
                action_batch[agent].append(actions[agent])
                reward_batch[agent].append(rewards[agent])
                next_obs_batch[agent].append(next_obs[agent])
                done_batch[agent].append(dones[agent])

        # 转换为张量
        processed_batch = {
            'obs': {
                agent: torch.as_tensor(np.array(obs_batch[agent]), 
                                     device=self.device, 
                                     dtype=torch.float32)
                for agent in obs_batch
            },
            'actions': {
                agent: torch.as_tensor(np.array(action_batch[agent]), 
                                     device=self.device, 
                                     dtype=torch.float32)
                for agent in action_batch
            },
            'rewards': {
                agent: torch.as_tensor(np.array(reward_batch[agent]), 
                                     device=self.device, 
                                     dtype=torch.float32).unsqueeze(1)
                for agent in reward_batch
            },
            'next_obs': {
                agent: torch.as_tensor(np.array(next_obs_batch[agent]), 
                                     device=self.device, 
                                     dtype=torch.float32)
                for agent in next_obs_batch
            },
            'dones': {
                agent: torch.as_tensor(np.array(done_batch[agent]), 
                                     device=self.device, 
                                     dtype=torch.bool).unsqueeze(1)
                for agent in done_batch
            }
        }

        return (processed_batch['obs'], 
                processed_batch['actions'],
                processed_batch['rewards'], 
                processed_batch['next_obs'],
                processed_batch['dones'],
                weights,
                indices)

    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        # 添加最大优先级限制
        priorities = np.clip((np.abs(td_errors) + 1e-5) ** self.alpha, 0, 1000.0)
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = min(max(self.max_priority, priority), 1000.0)

    def __len__(self):
        return self.tree.size

# 优化的MATD3实现
class MATD3(nn.Module):
    def __init__(self, agents, obs_dims, action_dims, actor_lr, critic_lr, gamma, tau, batch_size, 
                 policy_delay=2, noise_std=0.2, noise_clip=0.5, communication_radius=1.0):
        super(MATD3, self).__init__()
        self.agents = agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.total_it = 0
        self.communication_radius = communication_radius
        self._adj_cache = {}
        self._max_cache_size = 1000  # 限制缓存大小

        # 网络初始化
        self.actors = nn.ModuleDict()
        self.target_actors = nn.ModuleDict()
        self.critics1 = nn.ModuleDict()
        self.critics2 = nn.ModuleDict()
        self.target_critics1 = nn.ModuleDict()
        self.target_critics2 = nn.ModuleDict()

        total_obs_dim = sum(obs_dims.values())
        total_action_dim = sum(action_dims.values())
        n_agents = len(agents)

        # 使用 DataParallel 来并行化计算
        if torch.cuda.device_count() > 1:
            self.use_multi_gpu = True
        else:
            self.use_multi_gpu = False

        for agent in agents:
            # 演员网络
            self.actors[agent] = Actor(obs_dims[agent], action_dims[agent])
            self.target_actors[agent] = Actor(obs_dims[agent], action_dims[agent])
            self.target_actors[agent].load_state_dict(self.actors[agent].state_dict())

            # 评论家网络
            self.critics1[agent] = Critic(total_obs_dim, total_action_dim, n_agents=n_agents)
            self.critics2[agent] = Critic(total_obs_dim, total_action_dim, n_agents=n_agents)
            self.target_critics1[agent] = Critic(total_obs_dim, total_action_dim, n_agents=n_agents)
            self.target_critics2[agent] = Critic(total_obs_dim, total_action_dim, n_agents=n_agents)
            
            if self.use_multi_gpu:
                self.critics1[agent] = nn.DataParallel(self.critics1[agent])
                self.critics2[agent] = nn.DataParallel(self.critics2[agent])
                self.target_critics1[agent] = nn.DataParallel(self.target_critics1[agent])
                self.target_critics2[agent] = nn.DataParallel(self.target_critics2[agent])

            self.target_critics1[agent].load_state_dict(self.critics1[agent].state_dict())
            self.target_critics2[agent].load_state_dict(self.critics2[agent].state_dict())

        # 优化器初始化
        self.actor_optimizers = {agent: optim.Adam(self.actors[agent].parameters(), lr=actor_lr) for agent in agents}
        self.critic_optimizers1 = {agent: optim.Adam(self.critics1[agent].parameters(), lr=critic_lr) for agent in agents}
        self.critic_optimizers2 = {agent: optim.Adam(self.critics2[agent].parameters(), lr=critic_lr) for agent in agents}

    def _manage_cache(self):
        """管理缓存大小"""
        if len(self._adj_cache) > self._max_cache_size:
            # 删除最早的20%的缓存
            num_to_remove = int(self._max_cache_size * 0.2)
            keys_to_remove = list(self._adj_cache.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self._adj_cache[key]

    def _build_adjacency_matrix(self, obs):
        """构建邻接矩阵，使用缓存机制"""
        batch_size = obs[self.agents[0]].size(0)
        cache_key = f"{batch_size}"
        
        # 检查缓存
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key]
        
        n_agents = len(self.agents)
        adj_matrices = []
        
        # 使用向量化操作计算距离
        positions = torch.stack([obs[agent][:, :2] for agent in self.agents], dim=1)  # [batch_size, n_agents, 2]
        
        for b in range(batch_size):
            # 计算所有智能体之间的距离矩阵
            pos = positions[b]  # [n_agents, 2]
            distances = torch.cdist(pos, pos)  # [n_agents, n_agents]
            
            # 创建邻接矩阵
            adj_matrix = (distances <= self.communication_radius).float()
            adj_matrix.fill_diagonal_(1.0)  # 添加自环
            
            # 确保连通性
            if not self._is_connected(adj_matrix):
                adj_matrix = self._ensure_connectivity(adj_matrix)
            
            adj_matrices.append(adj_matrix)
        
        result = torch.stack(adj_matrices)
        
        # 更新缓存
        self._adj_cache[cache_key] = result
        self._manage_cache()
        
        return result

    @torch.no_grad()
    def _is_connected(self, adj_matrix):
        """使用广度优先搜索检查连通性，比矩阵幂更高效"""
        n = adj_matrix.size(0)
        visited = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
        queue = [0]  # 从第一个节点开始
        visited[0] = True
        
        while queue:
            node = queue.pop(0)
            neighbors = torch.where(adj_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor.item())
        
        return torch.all(visited)

    def _ensure_connectivity(self, adj_matrix):
        """确保图是连通的，如果不连通则添加必要的边"""
        if self._is_connected(adj_matrix):
            return adj_matrix
        
        # 复制邻接矩阵以避免修改原始数据
        new_adj = adj_matrix.clone()
        n = adj_matrix.size(0)
        
        # 使用最小生成树的思想连接所有节点
        unvisited = set(range(n))
        visited = {0}  # 从第一个节点开始
        unvisited.remove(0)
        
        while unvisited:
            min_dist = float('inf')
            edge = None
            
            # 找到最近的未访问节点
            for i in visited:
                for j in unvisited:
                    # 如果节点间没有连接，添加一条边
                    if new_adj[i, j] == 0:
                        new_adj[i, j] = 1
                        new_adj[j, i] = 1
                        edge = (i, j)
                        break
                if edge:
                    break
                
            if edge:
                visited.add(edge[1])
                unvisited.remove(edge[1])
        
        # 确保对称性和自环
        new_adj = new_adj + new_adj.t()  # 确保对称
        new_adj.fill_diagonal_(1)  # 添加自环
        
        return new_adj

    def update(self, replay_buffer):
        self.total_it += 1
        
        if len(replay_buffer) < self.batch_size:
            print(f"Waiting for replay buffer to fill up. Current size: {len(replay_buffer)}/{self.batch_size}")
            return None, None

        # 批量数据预处理
        obs, actions, rewards, next_obs, dones, weights, indices = replay_buffer.sample()
        
        try:
            # 使用 torch.cuda.amp 进行混合精度训练
            with torch.cuda.amp.autocast():
                # 构建邻接矩阵（使用缓存）
                adj_matrices = self._build_adjacency_matrix(obs)
                next_adj_matrices = self._build_adjacency_matrix(next_obs)

                # 预计算全局观察和动作
                global_obs = torch.cat([obs[ag] for ag in self.agents], dim=-1)
                global_actions = torch.cat([actions[ag] for ag in self.agents], dim=-1)
                next_global_obs = torch.cat([next_obs[ag] for ag in self.agents], dim=-1)

                total_critic_loss = 0
                total_actor_loss = 0
                td_errors = {agent: [] for agent in self.agents}

                # 批量更新所有智能体
                for agent in self.agents:
                    # Critic update
                    current_Q1 = self.critics1[agent](global_obs, global_actions, adj_matrices)
                    current_Q2 = self.critics2[agent](global_obs, global_actions, adj_matrices)

                    with torch.no_grad():
                        # Target actions
                        target_actions = []
                        for ag in self.agents:
                            target_act = self.target_actors[ag](next_obs[ag])
                            noise = (torch.randn_like(target_act) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
                            target_act = (target_act + noise).clamp(-1, 1)
                            target_actions.append(target_act)
                        target_global_actions = torch.cat(target_actions, dim=-1)

                        # Target Q-values
                        target_Q1 = self.target_critics1[agent](next_global_obs, target_global_actions, next_adj_matrices)
                        target_Q2 = self.target_critics2[agent](next_global_obs, target_global_actions, next_adj_matrices)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = rewards[agent] + (1 - dones[agent].float()) * self.gamma * target_Q

                    # Compute critic loss
                    critic_loss1 = F.mse_loss(current_Q1, target_Q)
                    critic_loss2 = F.mse_loss(current_Q2, target_Q)

                    # Update critics
                    self.critic_optimizers1[agent].zero_grad()
                    self.critic_optimizers2[agent].zero_grad()
                    critic_loss1.backward()
                    critic_loss2.backward()
                    torch.nn.utils.clip_grad_norm_(self.critics1[agent].parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critics2[agent].parameters(), 1.0)
                    self.critic_optimizers1[agent].step()
                    self.critic_optimizers2[agent].step()

                    total_critic_loss += (critic_loss1 + critic_loss2).item()

                    # Delayed policy updates
                    if self.total_it % self.policy_delay == 0:
                        # Actor update
                        actor_actions = []
                        for ag in self.agents:
                            if ag == agent:
                                actor_actions.append(self.actors[ag](obs[ag]))
                            else:
                                actor_actions.append(actions[ag])
                        actor_global_actions = torch.cat(actor_actions, dim=-1)

                        actor_loss = -self.critics1[agent](global_obs, actor_global_actions, adj_matrices).mean()
                        
                        # Update actor
                        self.actor_optimizers[agent].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1.0)
                        self.actor_optimizers[agent].step()
                        
                        total_actor_loss += actor_loss.item()
                        
                        # Soft update target networks
                        self._soft_update(agent)

                # 计算平均损失
                avg_critic_loss = total_critic_loss / len(self.agents)
                avg_actor_loss = total_actor_loss / len(self.agents) if self.total_it % self.policy_delay == 0 else 0

                return avg_actor_loss, avg_critic_loss

        except Exception as e:
            print(f"Error in update: {e}")
            return None, None

    def _soft_update(self, agent):
        """软更新目标网络"""
        with torch.no_grad():
            for param, target_param in zip(self.actors[agent].parameters(), 
                                         self.target_actors[agent].parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.critics1[agent].parameters(), 
                                         self.target_critics1[agent].parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.critics2[agent].parameters(), 
                                         self.target_critics2[agent].parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

def next_obs_combined(next_obs, agents):
    # 辅助函数：将多个智能体的 next_obs 合并
    return torch.cat([next_obs[ag] for ag in agents], dim=-1)

class SumTree:
    """
    用于优先经验回放的二叉树数据结构
    - 叶节点存储优先级值
    - 内部节点存储子节点的和
    """
    def __init__(self, capacity):
        self.capacity = capacity  # 叶节点数量（经验数量）
        self.tree = np.zeros(2 * capacity - 1)  # 整棵树的节点数组
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.data_pointer = 0  # 当前要写入数据的位置
        self.size = 0  # 当前存储的经验数量

    def propagate(self, idx, change):
        """更新父节点的和"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        """
        找到对应优先级和的叶节点
        s: 优先级和的采样值
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # 到达叶节点
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def add(self, priority, data):
        """添加新数据和对应的优先级"""
        tree_idx = self.data_pointer + self.capacity - 1  # 转换为叶节点索引
        
        # 存储数据
        self.data[self.data_pointer] = data
        
        # 更新优先级
        self.update(tree_idx, priority)
        
        # 更新指针和大小
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        """更新某个叶节点的优先级"""
        # 确保索引在有效范围内
        if idx >= len(self.tree):
            idx = idx % len(self.tree)
            
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get(self, s):
        """
        根据优先级和的采样值获取数据
        返回: (tree_idx, priority, data)
        """
        idx = self.retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        # 确保data_idx在有效范围内
        data_idx = data_idx % self.capacity
        
        return (idx, self.tree[idx], self.data[data_idx])

    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]