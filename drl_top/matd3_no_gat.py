import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import random

# 设置cudnn基准模式加速计算
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Actor网络
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128]):
        super(Actor, self).__init__()
        layers = []
        prev_dim = obs_dim
        
        # 构建网络层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()  # 改用ReLU激活函数，计算更快
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
    def forward(self, obs):
        return self.net(obs)

# 简化的Critic网络（不使用GAT）
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128]):
        super(Critic, self).__init__()
        layers = []
        prev_dim = obs_dim + action_dim
        
        # 构建网络层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()  # 改用ReLU激活函数，计算更快
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

# 优化的Replay Buffer实现
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.position = 0
        
    def add(self, obs, actions, rewards, next_obs, dones):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, actions, rewards, next_obs, dones)
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        obs_batch = defaultdict(list)
        action_batch = defaultdict(list)
        reward_batch = defaultdict(list)
        next_obs_batch = defaultdict(list)
        done_batch = defaultdict(list)
        
        for obs, actions, rewards, next_obs, dones in batch:
            for agent in obs:
                obs_batch[agent].append(obs[agent])
                action_batch[agent].append(actions[agent])
                reward_batch[agent].append(rewards[agent])
                next_obs_batch[agent].append(next_obs[agent])
                done_batch[agent].append(dones[agent])
        
        # 使用预分配和批量转换提高性能
        return (
            {k: torch.FloatTensor(np.array(v)).to(self.device) for k, v in obs_batch.items()},
            {k: torch.FloatTensor(np.array(v)).to(self.device) for k, v in action_batch.items()},
            {k: torch.FloatTensor(np.array(v)).unsqueeze(1).to(self.device) for k, v in reward_batch.items()},
            {k: torch.FloatTensor(np.array(v)).to(self.device) for k, v in next_obs_batch.items()},
            {k: torch.BoolTensor(np.array(v)).unsqueeze(1).to(self.device) for k, v in done_batch.items()}
        )
        
    def __len__(self):
        return len(self.buffer)

# 优化的MATD3实现（不使用GAT）
class MATD3:
    def __init__(self, agents, obs_dims, action_dims, device,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        
        self.device = device
        self.agents = agents
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        
        # 初始化网络
        self.actors = {}
        self.critics_1 = {}
        self.critics_2 = {}
        self.target_actors = {}
        self.target_critics_1 = {}
        self.target_critics_2 = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        
        for agent in agents:
            # 创建网络
            self.actors[agent] = Actor(obs_dims[agent], action_dims[agent]).to(device)
            self.critics_1[agent] = Critic(obs_dims[agent], action_dims[agent]).to(device)
            self.critics_2[agent] = Critic(obs_dims[agent], action_dims[agent]).to(device)
            
            self.target_actors[agent] = Actor(obs_dims[agent], action_dims[agent]).to(device)
            self.target_critics_1[agent] = Critic(obs_dims[agent], action_dims[agent]).to(device)
            self.target_critics_2[agent] = Critic(obs_dims[agent], action_dims[agent]).to(device)
            
            # 复制参数
            self.target_actors[agent].load_state_dict(self.actors[agent].state_dict())
            self.target_critics_1[agent].load_state_dict(self.critics_1[agent].state_dict())
            self.target_critics_2[agent].load_state_dict(self.critics_2[agent].state_dict())
            
            # 创建优化器
            self.actor_optimizers[agent] = optim.Adam(self.actors[agent].parameters(), lr=actor_lr)
            critic_params = list(self.critics_1[agent].parameters()) + list(self.critics_2[agent].parameters())
            self.critic_optimizers[agent] = optim.Adam(critic_params, lr=critic_lr)
    
    @torch.no_grad()
    def select_action(self, obs, noise=0.1):
        actions = {}
        # 批量处理所有智能体的观测和动作计算
        obs_tensors = {}
        for agent in self.agents:
            if agent not in obs:
                continue
            obs_tensors[agent] = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
        
        # 批量计算动作
        for agent, state in obs_tensors.items():
            action = self.actors[agent](state).squeeze(0).cpu().numpy()
            if noise != 0:
                action = action + np.random.normal(0, noise, size=action.shape)
                action = np.clip(action, -1, 1)
            actions[agent] = action
        return actions
    
    def train(self, replay_buffer):
        self.total_it += 1
        
        obs, actions, rewards, next_obs, dones = replay_buffer.sample()
        
        actor_losses = []
        critic_losses = []
        
        # 更新每个智能体的网络
        for agent in self.agents:
            # 计算目标动作和Q值
            with torch.no_grad():
                # 目标动作
                next_action = self.target_actors[agent](next_obs[agent])
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-1, 1)
                
                # 目标Q值
                target_Q1 = self.target_critics_1[agent](next_obs[agent], next_action)
                target_Q2 = self.target_critics_2[agent](next_obs[agent], next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards[agent] + (1 - dones[agent].float()) * self.gamma * target_Q
            
            # 更新Critic
            current_Q1 = self.critics_1[agent](obs[agent], actions[agent])
            current_Q2 = self.critics_2[agent](obs[agent], actions[agent])
            
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_losses.append(critic_loss.item())
            
            self.critic_optimizers[agent].zero_grad(set_to_none=True)  # 使用set_to_none=True提高性能
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_1[agent].parameters(), 1.0)  # 梯度裁剪提高稳定性
            torch.nn.utils.clip_grad_norm_(self.critics_2[agent].parameters(), 1.0)
            self.critic_optimizers[agent].step()
            
            # 延迟策略更新
            if self.total_it % self.policy_delay == 0:
                # 更新Actor
                actor_loss = -self.critics_1[agent](obs[agent], self.actors[agent](obs[agent])).mean()
                actor_losses.append(actor_loss.item())
                
                self.actor_optimizers[agent].zero_grad(set_to_none=True)  # 使用set_to_none=True提高性能
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1.0)  # 梯度裁剪提高稳定性
                self.actor_optimizers[agent].step()
                
                # 软更新目标网络
                self._soft_update(agent)
        
        return {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'critic_loss': np.mean(critic_losses)
        }
    
    def _soft_update(self, agent):
        # 使用torch.no_grad()提高性能
        with torch.no_grad():
            # 软更新目标网络
            for param, target_param in zip(self.actors[agent].parameters(), self.target_actors[agent].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critics_1[agent].parameters(), self.target_critics_1[agent].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critics_2[agent].parameters(), self.target_critics_2[agent].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actors': {agent: model.state_dict() for agent, model in self.actors.items()},
            'critics_1': {agent: model.state_dict() for agent, model in self.critics_1.items()},
            'critics_2': {agent: model.state_dict() for agent, model in self.critics_2.items()},
            'target_actors': {agent: model.state_dict() for agent, model in self.target_actors.items()},
            'target_critics_1': {agent: model.state_dict() for agent, model in self.target_critics_1.items()},
            'target_critics_2': {agent: model.state_dict() for agent, model in self.target_critics_2.items()},
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        for agent in self.agents:
            self.actors[agent].load_state_dict(checkpoint['actors'][agent])
            self.critics_1[agent].load_state_dict(checkpoint['critics_1'][agent])
            self.critics_2[agent].load_state_dict(checkpoint['critics_2'][agent])
            self.target_actors[agent].load_state_dict(checkpoint['target_actors'][agent])
            self.target_critics_1[agent].load_state_dict(checkpoint['target_critics_1'][agent])
            self.target_critics_2[agent].load_state_dict(checkpoint['target_critics_2'][agent]) 