import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import random

# 设置cudnn基准模式加速卷积
torch.backends.cudnn.benchmark = True

# Actor网络
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
        
        # 添加残差连接
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

# 简化的Critic网络（不使用GAT）
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # 第一层处理观测和动作的组合
        prev_dim = obs_dim + action_dim
        
        # 动态构建隐藏层
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
            
        # 输出层
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 添加残差连接
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
            
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        
        # 第一层特殊处理
        x = F.leaky_relu(self.layer_norms[0](self.layers[0](x)), negative_slope=0.01)
        
        # 处理中间层，添加残差连接
        for i in range(1, len(self.layers)):
            identity = x
            x = self.layers[i](x)
            x = self.layer_norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.01)
            
            # 添加残差连接
            if i-1 < len(self.residual_layers):
                residual = self.residual_layers[i-1](identity)
                x = x + residual
        
        return self.output_layer(x)

# 优化的Replay Buffer实现
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
        """添加新的经验到缓冲区"""
        experience = (obs, actions, rewards, next_obs, dones)
        self.tree.add(self.max_priority, experience)

    def sample(self):
        """采样一个批次的经验"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / self.batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        priorities = np.array(priorities)
        weights = self._compute_weights(priorities)

        return self._process_batch(batch, indices, weights)

    def _compute_weights(self, priorities):
        """计算重要性采样权重"""
        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-self.beta)
        weights = weights / weights.max()
        return torch.as_tensor(weights, device=self.device, dtype=torch.float32)

    def _process_batch(self, batch, indices, weights):
        """处理批量数据"""
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
        priorities = np.clip((np.abs(td_errors) + 1e-5) ** self.alpha, 0, 1000.0)
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = min(max(self.max_priority, priority), 1000.0)

    def __len__(self):
        return self.tree.size

# 优化的MATD3实现（不使用GAT）
class MATD3(nn.Module):
    def __init__(self, agents, obs_dims, action_dims, actor_lr, critic_lr, gamma, tau, batch_size, 
                 policy_delay=2, noise_std=0.2, noise_clip=0.5):
        super(MATD3, self).__init__()
        self.agents = agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.total_it = 0

        # 网络初始化
        self.actors = nn.ModuleDict()
        self.target_actors = nn.ModuleDict()
        self.critics1 = nn.ModuleDict()
        self.critics2 = nn.ModuleDict()
        self.target_critics1 = nn.ModuleDict()
        self.target_critics2 = nn.ModuleDict()

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
            self.critics1[agent] = Critic(obs_dims[agent], action_dims[agent])
            self.critics2[agent] = Critic(obs_dims[agent], action_dims[agent])
            self.target_critics1[agent] = Critic(obs_dims[agent], action_dims[agent])
            self.target_critics2[agent] = Critic(obs_dims[agent], action_dims[agent])
            
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

    def update(self, replay_buffer):
        self.total_it += 1
        
        if len(replay_buffer) < self.batch_size:
            return None, None

        # 批量数据预处理
        obs, actions, rewards, next_obs, dones, weights, indices = replay_buffer.sample()
        
        try:
            with torch.cuda.amp.autocast():
                total_critic_loss = 0
                total_actor_loss = 0
                td_errors = {agent: [] for agent in self.agents}

                # 批量更新所有智能体
                for agent in self.agents:
                    # Critic update
                    current_Q1 = self.critics1[agent](obs[agent], actions[agent])
                    current_Q2 = self.critics2[agent](obs[agent], actions[agent])

                    with torch.no_grad():
                        # Target actions with noise
                        target_action = self.target_actors[agent](next_obs[agent])
                        noise = (torch.randn_like(target_action) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
                        target_action = (target_action + noise).clamp(-1, 1)

                        # Target Q-values
                        target_Q1 = self.target_critics1[agent](next_obs[agent], target_action)
                        target_Q2 = self.target_critics2[agent](next_obs[agent], target_action)
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
                        actor_loss = -self.critics1[agent](obs[agent], self.actors[agent](obs[agent])).mean()
                        
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

class SumTree:
    """用于优先经验回放的二叉树数据结构"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def propagate(self, idx, change):
        """更新父节点的和"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        """找到对应优先级和的叶节点"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def add(self, priority, data):
        """添加新数据和对应的优先级"""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        """更新某个叶节点的优先级"""
        if idx >= len(self.tree):
            idx = idx % len(self.tree)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get(self, s):
        """根据优先级和的采样值获取数据"""
        idx = self.retrieve(0, s)
        data_idx = idx - self.capacity + 1
        data_idx = data_idx % self.capacity
        return (idx, self.tree[idx], self.data[data_idx])

    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]