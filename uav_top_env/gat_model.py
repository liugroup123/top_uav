import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class UAVAttentionNetwork(nn.Module):
    def __init__(self, uav_features, target_features, hidden_size=64, heads=4, dropout=0.6, device=None):
        super(UAVAttentionNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # UAV-UAV 交互
        self.uav_gat1 = GATConv(uav_features, hidden_size, heads=heads, dropout=dropout).to(self.device)
        self.uav_gat2 = GATConv(hidden_size * heads, hidden_size, heads=1, dropout=dropout).to(self.device)
        
        # 目标转换层
        self.target_transform = nn.Linear(target_features, hidden_size).to(self.device)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2)  # 输出特征维度可调
        ).to(self.device)

    def forward(self, uav_features, target_features, uav_adj, target_adj):
        # UAV特征提取
        uav_edge_index = adj_matrix_to_edge_index(uav_adj)
        uav_h = F.elu(self.uav_gat1(uav_features, uav_edge_index))
        uav_h = self.uav_gat2(uav_h, uav_edge_index)
        
        # 目标特征提取
        target_h = self.target_transform(target_features)
        
        # 为每个UAV提取相关目标特征
        target_features = []
        for i in range(len(uav_features)):
            visible_targets = target_h[target_adj[i] > 0]
            if len(visible_targets) > 0:
                target_feat = torch.mean(visible_targets, dim=0)
            else:
                target_feat = torch.zeros(uav_h.size(-1), device=self.device)
            target_features.append(target_feat)
        
        target_features = torch.stack(target_features)
        
        # 特征融合
        combined = torch.cat([uav_h, target_features], dim=-1)
        gat_features = self.fusion_layer(combined)
        
        return gat_features

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

def create_adjacency_matrices(uav_positions, target_positions, comm_radius, coverage_radius):
    n_uavs = len(uav_positions)
    n_targets = len(target_positions)
    device = uav_positions.device
    
    # UAV-UAV邻接矩阵
    uav_adj = torch.zeros((n_uavs, n_uavs), device=device)
    for i in range(n_uavs):
        for j in range(n_uavs):
            if i != j:
                dist = torch.norm(uav_positions[i] - uav_positions[j])
                if dist <= comm_radius:
                    uav_adj[i][j] = 1.0
                    
    # UAV-Target邻接矩阵
    target_adj = torch.zeros((n_uavs, n_targets), device=device)
    for i in range(n_uavs):
        for j in range(n_targets):
            dist = torch.norm(uav_positions[i] - target_positions[j])
            if dist <= coverage_radius:
                target_adj[i][j] = 1.0
                
    return uav_adj, target_adj
