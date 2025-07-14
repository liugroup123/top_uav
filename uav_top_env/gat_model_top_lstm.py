import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.cuda.amp import autocast

# 启用CUDA优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class UAVAttentionNetwork(nn.Module):
    def __init__(self, uav_features, target_features, hidden_size=64, heads=4, dropout=0.6, device=None, 
                 lstm_hidden_size=64, lstm_layers=1):
        super(UAVAttentionNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training = True  # 添加训练标志
        
        # LSTM相关参数
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.hidden_states = {}  # 存储各智能体的LSTM隐藏状态
        self.update_states_in_inference = True  # 推理模式下是否更新状态
        
        # 梯度累积参数
        self.grad_accumulation_steps = 4  # 累积4个批次的梯度再更新
        self.current_step = 0
        
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
            ),
            # 新增：LSTM模块 - 使用cuDNN优化版本
            'lstm': nn.LSTM(
                input_size=hidden_size//2,  # 输入是融合层输出的特征维度
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0
            ),
            # 新增：LSTM输出映射层
            'lstm_output': nn.Linear(lstm_hidden_size, hidden_size//2)
        }).to(self.device)
        
        # 确保LSTM使用cuDNN优化
        if self.device.type == 'cuda':
            # 设置为False可以在序列长度变化时获得更好的性能
            torch.backends.cudnn.deterministic = False

    def reset_lstm_states(self):
        """重置所有LSTM隐藏状态"""
        self.hidden_states = {}
        
    def set_inference_mode(self):
        """设置为推理模式但保持状态更新"""
        self.eval()  # 切换到评估模式
        self.update_states_in_inference = True  # 确保推理时仍更新隐藏状态

    @autocast()  # 使用混合精度训练，不需要指定device_type参数
    def forward(self, uav_features, target_features, uav_adj, target_adj, 
                active_agents=None, agent_ids=None, reset_states=False):
        """
        前向传播，支持训练和评估模式，增加LSTM处理
        Args:
            uav_features: UAV的特征
            target_features: 目标的特征
            uav_adj: UAV间的邻接矩阵
            target_adj: UAV-目标间的邻接矩阵
            active_agents: 活跃UAV的索引列表
            agent_ids: 智能体ID列表，用于LSTM状态管理
            reset_states: 是否重置LSTM状态
        """
        # 优化4：确保输入数据在正确的设备上，使用non_blocking=True加速传输
        uav_features = uav_features.to(self.device, non_blocking=True)
        target_features = target_features.to(self.device, non_blocking=True)
        uav_adj = uav_adj.to(self.device, non_blocking=True)
        target_adj = target_adj.to(self.device, non_blocking=True)
        
        if active_agents is None:
            active_agents = list(range(len(uav_features)))
            
        if agent_ids is None:
            agent_ids = list(range(len(uav_features)))
        
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
        # 创建掩码张量，用于批量处理
        mask = torch.zeros(len(uav_features), device=self.device)
        mask[active_agents] = 1
        
        # 批量计算目标特征
        target_features_batch = []
        for i in range(len(uav_features)):
            if mask[i]:
                visible_mask = target_adj[i] > 0
                if visible_mask.any():
                    target_feat = target_h[visible_mask].mean(dim=0)
                else:
                    target_feat = torch.zeros(uav_h.size(-1), device=self.device)
            else:
                target_feat = torch.zeros(uav_h.size(-1), device=self.device)
            target_features_batch.append(target_feat)
        
        # 使用stack一次性处理所有特征
        target_features = torch.stack(target_features_batch)
        
        # 特征融合 - GAT输出
        combined = torch.cat([uav_h, target_features], dim=-1)
        gat_output = self.model['fusion_layer'](combined)
        
        # LSTM处理 - 批量处理时序信息
        # 将所有智能体的特征合并为一个批次
        batch_size = len(agent_ids)
        
        # 准备批量处理的输入
        lstm_batch_input = gat_output.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 准备隐藏状态
        if reset_states or not self.hidden_states:
            # 批量初始化隐藏状态
            h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=self.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=self.device)
            hidden_batch = (h0, c0)
        else:
            # 从现有状态中收集批次的隐藏状态
            h_list = []
            c_list = []
            
            for agent_id in agent_ids:
                if agent_id in self.hidden_states:
                    h, c = self.hidden_states[agent_id]
                else:
                    h = torch.zeros(self.lstm_layers, 1, self.lstm_hidden_size, device=self.device)
                    c = torch.zeros(self.lstm_layers, 1, self.lstm_hidden_size, device=self.device)
                h_list.append(h)
                c_list.append(c)
            
            # 合并为批次
            h_batch = torch.cat([h.transpose(0, 1) for h in h_list], dim=0).transpose(0, 1)
            c_batch = torch.cat([c.transpose(0, 1) for c in c_list], dim=0).transpose(0, 1)
            hidden_batch = (h_batch, c_batch)
        
        # 批量LSTM前向传播
        lstm_out, new_hidden_batch = self.model['lstm'](lstm_batch_input, hidden_batch)
        
        # 更新隐藏状态
        if self.training or self.update_states_in_inference:
            h_new, c_new = new_hidden_batch
            for idx, agent_id in enumerate(agent_ids):
                # 为每个智能体提取并保存其隐藏状态
                h_agent = h_new[:, idx:idx+1, :]
                c_agent = c_new[:, idx:idx+1, :]
                self.hidden_states[agent_id] = (h_agent, c_agent)
        
        # 处理LSTM输出
        lstm_output = self.model['lstm_output'](lstm_out.squeeze(1))
        
        # 增加梯度累积步数计数
        if self.training:
            self.current_step = (self.current_step + 1) % self.grad_accumulation_steps
        
        return lstm_output

    def should_update_weights(self):
        """检查是否应该更新权重（用于梯度累积）"""
        return self.current_step == 0

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
    
    # 批量计算UAV-UAV距离矩阵
    uav_pos_expanded = uav_positions.unsqueeze(1)  # [n_uavs, 1, 2]
    uav_pos_expanded2 = uav_positions.unsqueeze(0)  # [1, n_uavs, 2]
    uav_dist_matrix = torch.norm(uav_pos_expanded - uav_pos_expanded2, dim=2)  # [n_uavs, n_uavs]
    
    # 创建UAV-UAV邻接矩阵
    uav_adj = (uav_dist_matrix <= comm_radius).float()
    # 移除自环
    uav_adj.fill_diagonal_(0)
    
    # 创建掩码，只保留活跃UAV的连接
    active_mask = torch.zeros((n_uavs, n_uavs), device=device, dtype=torch.bool)
    for i in active_uavs:
        for j in active_uavs:
            active_mask[i, j] = True
    
    uav_adj = uav_adj * active_mask.float()
    
    # 批量计算UAV-Target距离矩阵
    uav_pos_expanded = uav_positions.unsqueeze(1)  # [n_uavs, 1, 2]
    target_pos_expanded = target_positions.unsqueeze(0)  # [1, n_targets, 2]
    target_dist_matrix = torch.norm(uav_pos_expanded - target_pos_expanded, dim=2)  # [n_uavs, n_targets]
    
    # 创建UAV-Target邻接矩阵
    target_adj = (target_dist_matrix <= coverage_radius).float()
    
    # 应用活跃UAV掩码
    active_uav_mask = torch.zeros(n_uavs, device=device)
    active_uav_mask[active_uavs] = 1.0
    target_adj = target_adj * active_uav_mask.unsqueeze(1)
    
    return uav_adj, target_adj