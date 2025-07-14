import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class UAVAttentionNetwork(nn.Module):
    def __init__(self, uav_features, target_features, hidden_size=64, heads=4, dropout=0.6, device=None):
        super(UAVAttentionNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training = True  # 添加训练标志
        
        # 优化2：一次性将整个模型移到GPU
        self.model = nn.ModuleDict({
            'uav_gat1': GATConv(
                uav_features, 
                hidden_size, 
                heads=heads, 
                dropout=dropout, 
                add_self_loops=True,
                concat=True  # 连接多头注意力的输出
            ),
            'uav_gat2': GATConv(
                hidden_size * heads, 
                hidden_size, 
                heads=1, 
                dropout=dropout, 
                add_self_loops=True,
                concat=False  # 最后一层不连接多头输出
            ),
            'bn1': nn.BatchNorm1d(hidden_size * heads),
            'bn2': nn.BatchNorm1d(hidden_size),
            'target_transform': nn.Linear(target_features, hidden_size),
            'target_bn': nn.BatchNorm1d(hidden_size),
            'fusion_layer': nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size//2)
            )
        }).to(self.device)

    @torch.cuda.amp.autocast()  # 优化3：使用混合精度训练
    def forward(self, uav_features, target_features, uav_adj, target_adj, active_agents=None):
        """
        前向传播，支持训练和评估模式
        Args:
            uav_features: UAV的特征
            target_features: 目标的特征
            uav_adj: UAV间的邻接矩阵
            target_adj: UAV-目标间的邻接矩阵
            active_agents: 活跃UAV的索引列表（可选）
        """
        # 优化4：确保输入数据在正确的设备上
        uav_features = uav_features.to(self.device)
        target_features = target_features.to(self.device)
        uav_adj = uav_adj.to(self.device)
        target_adj = target_adj.to(self.device)
        
        if active_agents is None:
            active_agents = list(range(len(uav_features)))
        
        # 优化5：批量处理
        uav_edge_index = adj_matrix_to_edge_index(uav_adj)
        
        # GAT处理
        x = self.model['uav_gat1'](uav_features, uav_edge_index)
        if x.size(0) > 1:  # 优化6：使用size(0)替代len()
            x = self.model['bn1'](x)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        
        x = self.model['uav_gat2'](x, uav_edge_index)
        if x.size(0) > 1:
            x = self.model['bn2'](x)
        uav_h = F.elu(x)
        
        # 优化7：并行处理目标特征
        target_h = self.model['target_transform'](target_features)
        if target_h.size(0) > 1:
            target_h = self.model['target_bn'](target_h)
        target_h = F.relu(target_h)
        
        # 优化8：使用向量化操作替代循环
        target_features_list = []
        mask = torch.zeros(len(uav_features), device=self.device)
        mask[active_agents] = 1
        
        for i in range(len(uav_features)):
            if mask[i]:
                visible_mask = target_adj[i] > 0
                if visible_mask.any():
                    target_feat = target_h[visible_mask].mean(dim=0)
                else:
                    target_feat = torch.zeros(uav_h.size(-1), device=self.device)
            else:
                target_feat = torch.zeros(uav_h.size(-1), device=self.device)
            target_features_list.append(target_feat)
        
        target_features = torch.stack(target_features_list)
        
        # 特征融合
        combined = torch.cat([uav_h, target_features], dim=-1)
        return self.model['fusion_layer'](combined)

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

def create_adjacency_matrices(uav_positions, target_positions, comm_radius, coverage_radius, active_uavs=None):
    """
    创建考虑活跃UAV的邻接矩阵
    
    Args:
        uav_positions: UAV位置
        target_positions: 目标位置
        comm_radius: 通信半径
        coverage_radius: 覆盖半径
        active_uavs: 活跃UAV的索引列表，如果为None则认为所有UAV都是活跃的
    """
    n_uavs = len(uav_positions)
    n_targets = len(target_positions)
    device = uav_positions.device
    
    if active_uavs is None:
        active_uavs = list(range(n_uavs))
    
    # UAV-UAV邻接矩阵
    uav_adj = torch.zeros((n_uavs, n_uavs), device=device)
    for i in active_uavs:
        for j in active_uavs:
            if i != j:
                dist = torch.norm(uav_positions[i] - uav_positions[j])
                if dist <= comm_radius:
                    uav_adj[i][j] = 1.0
    
    # UAV-Target邻接矩阵
    target_adj = torch.zeros((n_uavs, n_targets), device=device)
    for i in active_uavs:
        for j in range(n_targets):
            dist = torch.norm(uav_positions[i] - target_positions[j])
            if dist <= coverage_radius:
                target_adj[i][j] = 1.0
    
    return uav_adj, target_adj