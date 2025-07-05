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
        
        # 双GAT架构：UAV-UAV GAT + UAV-Target GAT
        self.model = nn.ModuleDict({
            # UAV-UAV GAT网络
            'uav_gat1': GATConv(
                uav_features,
                hidden_size,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True
            ),
            'uav_gat2': GATConv(
                hidden_size * heads,
                hidden_size,
                heads=1,
                dropout=dropout,
                add_self_loops=True,
                concat=False
            ),

            # UAV-Target GAT网络 (新增)
            'uav_target_gat1': GATConv(
                uav_features,  # UAV特征作为查询
                hidden_size,
                heads=heads,
                dropout=dropout,
                add_self_loops=False,  # UAV-Target不需要自循环
                concat=True
            ),
            'uav_target_gat2': GATConv(
                hidden_size * heads,
                hidden_size,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
                concat=False
            ),

            # 批归一化层
            'uav_bn1': nn.BatchNorm1d(hidden_size * heads),
            'uav_bn2': nn.BatchNorm1d(hidden_size),
            'target_bn1': nn.BatchNorm1d(hidden_size * heads),
            'target_bn2': nn.BatchNorm1d(hidden_size),

            # 目标特征预处理
            'target_transform': nn.Linear(target_features, uav_features),  # 将目标特征转换为UAV特征维度

            # 特征融合层
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
        
        # 1. UAV-UAV GAT处理
        uav_edge_index = adj_matrix_to_edge_index(uav_adj)

        # UAV-UAV GAT第一层
        uav_x = self.model['uav_gat1'](uav_features, uav_edge_index)
        if uav_x.size(0) > 1:
            uav_x = self.model['uav_bn1'](uav_x)
        uav_x = F.elu(uav_x)
        uav_x = F.dropout(uav_x, p=0.6, training=self.training)

        # UAV-UAV GAT第二层
        uav_x = self.model['uav_gat2'](uav_x, uav_edge_index)
        if uav_x.size(0) > 1:
            uav_x = self.model['uav_bn2'](uav_x)
        uav_h = F.elu(uav_x)

        # 2. UAV-Target GAT处理
        # 预处理目标特征，使其与UAV特征维度一致
        target_features_processed = self.model['target_transform'](target_features)

        # 创建UAV-Target二分图的边索引
        uav_target_edge_index = self._create_bipartite_edge_index(target_adj, active_agents)

        # 合并UAV和目标特征用于二分图GAT
        combined_features = torch.cat([uav_features, target_features_processed], dim=0)

        # UAV-Target GAT第一层
        target_x = self.model['uav_target_gat1'](combined_features, uav_target_edge_index)
        if target_x.size(0) > 1:
            target_x = self.model['target_bn1'](target_x)
        target_x = F.elu(target_x)
        target_x = F.dropout(target_x, p=0.6, training=self.training)

        # UAV-Target GAT第二层
        target_x = self.model['uav_target_gat2'](target_x, uav_target_edge_index)
        if target_x.size(0) > 1:
            target_x = self.model['target_bn2'](target_x)
        target_h = F.elu(target_x)

        # 只取UAV部分的特征（前n_uavs个节点）
        n_uavs = len(uav_features)
        uav_target_features = target_h[:n_uavs]

        # 3. 特征融合
        combined = torch.cat([uav_h, uav_target_features], dim=-1)
        return self.model['fusion_layer'](combined)

    def _create_bipartite_edge_index(self, target_adj, active_agents):
        """创建UAV-Target二分图的边索引"""
        n_uavs = target_adj.size(0)
        n_targets = target_adj.size(1)

        edges = []
        # UAV到Target的边
        for i in active_agents:
            for j in range(n_targets):
                if target_adj[i, j] > 0:
                    edges.append([i, n_uavs + j])  # 目标节点索引偏移n_uavs

        # Target到UAV的边（双向）
        for i in active_agents:
            for j in range(n_targets):
                if target_adj[i, j] > 0:
                    edges.append([n_uavs + j, i])

        if len(edges) == 0:
            # 如果没有边，创建空的边索引
            return torch.zeros((2, 0), dtype=torch.long, device=target_adj.device)

        edge_index = torch.tensor(edges, device=target_adj.device).t().contiguous()
        return edge_index

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