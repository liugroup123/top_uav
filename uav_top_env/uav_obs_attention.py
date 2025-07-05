# -*- coding: utf-8 -*-
"""
UAV观察空间Attention网络
在GAT处理后的观察基础上，进行UAV间的attention信息交换
支持多跳信息传播和动态注意力权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UAVObservationAttention(nn.Module):
    """
    UAV观察空间Attention网络
    在每个UAV的观察特征基础上，通过attention机制实现UAV间信息交换
    """
    
    def __init__(self, obs_dim=67, hidden_dim=64, num_heads=4, num_hops=2, dropout=0.1,
                 sparse_attention=True, max_neighbors=3, device='cuda'):
        super(UAVObservationAttention, self).__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hops = num_hops
        self.sparse_attention = sparse_attention
        self.max_neighbors = max_neighbors
        self.device = device
        
        # 观察特征编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_hops)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_hops)
        ])
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 输出解码器
        self.obs_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # 门控机制 - 控制原始观察和增强观察的融合
        self.gate = nn.Sequential(
            nn.Linear(obs_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, observations, communication_mask, active_agents=None):
        """
        前向传播
        
        Args:
            observations: [num_uavs, obs_dim] 每个UAV的观察
            communication_mask: [num_uavs, num_uavs] 通信掩码 (True表示可通信)
            active_agents: list 活跃UAV列表
            
        Returns:
            enhanced_observations: [num_uavs, obs_dim] 增强后的观察
            attention_weights: list 每一跳的注意力权重
        """
        num_uavs = observations.shape[0]
        
        # 处理活跃UAV
        if active_agents is None:
            active_agents = list(range(num_uavs))
        
        # 编码观察特征
        encoded_obs = self.obs_encoder(observations)  # [num_uavs, hidden_dim]
        
        # 添加batch维度用于attention计算
        encoded_obs = encoded_obs.unsqueeze(0)  # [1, num_uavs, hidden_dim]
        
        # 创建attention mask (True表示被mask，即不能attend)
        attention_mask = ~communication_mask  # [num_uavs, num_uavs]

        # 对非活跃UAV进行mask
        for i in range(num_uavs):
            if i not in active_agents:
                attention_mask[i, :] = True  # 非活跃UAV不能接收信息
                attention_mask[:, i] = True  # 非活跃UAV不能发送信息

        # 稀疏attention优化：限制每个UAV的最大邻居数
        if self.sparse_attention:
            attention_mask = self._create_sparse_mask(attention_mask, communication_mask, active_agents)
        
        current_obs = encoded_obs
        attention_weights_list = []
        
        # 多跳attention处理
        for hop in range(self.num_hops):
            # Self-attention
            attended_obs, attention_weights = self.attention_layers[hop](
                query=current_obs,
                key=current_obs,
                value=current_obs,
                attn_mask=attention_mask,
                need_weights=True
            )
            
            attention_weights_list.append(attention_weights.squeeze(0))  # 移除batch维度
            
            # 残差连接 + 层归一化
            current_obs = self.layer_norms[hop](current_obs + attended_obs)
            
            # 前馈网络 (只在最后一跳应用)
            if hop == self.num_hops - 1:
                ff_output = self.feed_forward(current_obs)
                current_obs = current_obs + ff_output
        
        # 移除batch维度
        enhanced_features = current_obs.squeeze(0)  # [num_uavs, hidden_dim]
        
        # 解码回观察空间
        enhanced_obs = self.obs_decoder(enhanced_features)  # [num_uavs, obs_dim]
        
        # 门控融合原始观察和增强观察
        gate_input = torch.cat([observations, enhanced_obs], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 最终观察 = gate * enhanced + (1-gate) * original
        final_observations = gate_weights * enhanced_obs + (1 - gate_weights) * observations
        
        return final_observations, attention_weights_list
    
    def get_attention_summary(self, attention_weights_list, active_agents):
        """
        获取注意力权重摘要，用于分析和可视化
        
        Args:
            attention_weights_list: 每一跳的注意力权重
            active_agents: 活跃UAV列表
            
        Returns:
            dict: 注意力摘要信息
        """
        summary = {
            'num_hops': len(attention_weights_list),
            'active_agents': active_agents,
            'attention_patterns': []
        }
        
        for hop, weights in enumerate(attention_weights_list):
            # weights: [num_uavs, num_uavs]
            hop_summary = {
                'hop': hop,
                'max_attention': weights.max().item(),
                'min_attention': weights.min().item(),
                'mean_attention': weights.mean().item(),
                'attention_entropy': self._compute_attention_entropy(weights, active_agents)
            }
            summary['attention_patterns'].append(hop_summary)
        
        return summary
    
    def _compute_attention_entropy(self, attention_weights, active_agents):
        """计算注意力分布的熵，衡量注意力的分散程度"""
        entropies = []
        for i in active_agents:
            # 获取UAV i对其他UAV的注意力分布
            attn_dist = attention_weights[i, active_agents]
            # 计算熵
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
            entropies.append(entropy.item())
        
        return np.mean(entropies) if entropies else 0.0

    def _create_sparse_mask(self, attention_mask, communication_mask, active_agents):
        """
        创建稀疏attention掩码，限制每个UAV的最大邻居数

        Args:
            attention_mask: [num_uavs, num_uavs] 原始attention掩码
            communication_mask: [num_uavs, num_uavs] 通信掩码
            active_agents: list 活跃UAV列表

        Returns:
            torch.Tensor: 稀疏化的attention掩码
        """
        sparse_mask = attention_mask.clone()

        for i in active_agents:
            # 找到UAV i可以通信的邻居
            neighbors = []
            for j in active_agents:
                if i != j and communication_mask[i, j]:
                    neighbors.append(j)

            # 如果邻居数超过最大限制，只保留最近的几个
            if len(neighbors) > self.max_neighbors:
                # 这里简化处理，实际可以根据距离排序
                # mask掉多余的邻居
                for j in neighbors[self.max_neighbors:]:
                    sparse_mask[i, j] = True

        return sparse_mask


def create_communication_mask(agent_positions, active_agents, communication_range, device='cuda'):
    """
    创建通信掩码矩阵
    
    Args:
        agent_positions: [num_uavs, 2] UAV位置
        active_agents: list 活跃UAV列表  
        communication_range: float 通信半径
        device: str 设备
        
    Returns:
        torch.Tensor: [num_uavs, num_uavs] 通信掩码 (True表示可通信)
    """
    num_uavs = len(agent_positions)
    comm_mask = torch.zeros((num_uavs, num_uavs), dtype=torch.bool, device=device)
    
    for i in active_agents:
        for j in active_agents:
            if i != j:
                dist = torch.norm(agent_positions[i] - agent_positions[j])
                if dist <= communication_range:
                    comm_mask[i, j] = True
    
    return comm_mask


# 测试函数
def test_uav_obs_attention():
    """测试UAV观察attention网络"""
    print("🧪 测试UAV观察Attention网络...")
    
    # 创建测试数据
    num_uavs = 5
    obs_dim = 67
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建网络
    attention_net = UAVObservationAttention(
        obs_dim=obs_dim,
        hidden_dim=64,
        num_heads=2,
        num_hops=2,
        device=device
    )
    
    # 创建测试观察
    observations = torch.randn(num_uavs, obs_dim, device=device)
    
    # 创建通信掩码 (假设所有UAV都能相互通信)
    comm_mask = torch.ones((num_uavs, num_uavs), dtype=torch.bool, device=device)
    comm_mask.fill_diagonal_(False)  # 自己不与自己通信
    
    active_agents = [0, 1, 2, 3, 4]
    
    # 前向传播
    enhanced_obs, attention_weights = attention_net(observations, comm_mask, active_agents)
    
    print(f"✅ 输入观察形状: {observations.shape}")
    print(f"✅ 输出观察形状: {enhanced_obs.shape}")
    print(f"✅ 注意力权重数量: {len(attention_weights)}")
    print(f"✅ 每个注意力权重形状: {attention_weights[0].shape}")
    
    # 获取注意力摘要
    summary = attention_net.get_attention_summary(attention_weights, active_agents)
    print(f"✅ 注意力摘要: {summary}")
    
    print("🎉 UAV观察Attention网络测试完成！")


if __name__ == "__main__":
    test_uav_obs_attention()
