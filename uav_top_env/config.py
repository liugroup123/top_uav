"""
UAV拓扑环境配置文件
支持多种实验配置和参数管理
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class EnvironmentConfig:
    """环境基础配置"""
    num_agents: int = 5
    num_targets: int = 10
    world_size: float = 1.0
    coverage_radius: float = 0.3
    communication_radius: float = 0.6
    max_steps: int = 200
    dt: float = 0.1
    
    # 渲染配置
    render_mode: Optional[str] = None  # 'human', 'rgb_array', None
    screen_size: tuple = (700, 700)
    render_fps: int = 60

@dataclass
class TopologyConfig:
    """拓扑变化配置"""
    experiment_type: str = 'normal'  # 'normal', 'probabilistic'
    
    # 变化控制参数
    topology_change_interval: Optional[int] = None  # 固定间隔模式的步数
    topology_change_probability: Optional[float] = None  # 随机模式的概率
    
    # UAV数量控制
    min_active_agents: Optional[int] = None  # 最少UAV数量
    max_active_agents: Optional[int] = None  # 最多UAV数量
    initial_active_ratio: Optional[float] = None  # 初始活跃比例(仅用于addition模式)

@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 覆盖率相关
    coverage_weight: float = 3.5
    coverage_exp: float = 1.5
    distance_scale: float = 1.5
    
    # 连通性相关
    connectivity_weight: float = 2.0
    min_connectivity_ratio: float = 0.3
    
    # 稳定性相关
    stability_weight: float = 1.5
    stability_time_window: int = 10
    stability_threshold: float = 0.9
    
    # 能量效率相关
    energy_weight: float = 0.5
    smoothness_weight: float = 0.5
    
    # 边界惩罚相关
    boundary_weight: float = 1.0
    safe_distance: float = 0.1
    
    # 拓扑变化相关
    reorganization_weight: float = 2.0
    task_reassign_weight: float = 1.5
    formation_weight: float = 1.0

@dataclass
class PhysicsConfig:
    """物理参数配置"""
    max_accel: float = 1.5
    max_speed: float = 2.0
    boundary_decay_zone: float = 0.2
    min_speed_ratio: float = 0.2
    bounce_energy_loss: float = 0.7

@dataclass
class GATConfig:
    """GAT网络配置"""
    uav_features: int = 4
    target_features: int = 2
    hidden_size: int = 64
    heads: int = 4
    dropout: float = 0.6

@dataclass
class ExperimentConfig:
    """完整实验配置"""
    name: str = "default_experiment"
    description: str = "默认实验配置"
    
    # 各模块配置
    environment: EnvironmentConfig = None
    topology: TopologyConfig = None
    reward: RewardConfig = None
    physics: PhysicsConfig = None
    gat: GATConfig = None
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.topology is None:
            self.topology = TopologyConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.gat is None:
            self.gat = GATConfig()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: ExperimentConfig, filename: str = None):
        """保存配置到文件"""
        if filename is None:
            filename = f"{config.name}.json"
        
        filepath = os.path.join(self.config_dir, filename)
        config_dict = asdict(config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到: {filepath}")
    
    def load_config(self, filename: str) -> ExperimentConfig:
        """从文件加载配置"""
        filepath = os.path.join(self.config_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 重构配置对象
        config = ExperimentConfig(
            name=config_dict.get('name', 'loaded_config'),
            description=config_dict.get('description', ''),
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            topology=TopologyConfig(**config_dict.get('topology', {})),
            reward=RewardConfig(**config_dict.get('reward', {})),
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            gat=GATConfig(**config_dict.get('gat', {}))
        )
        
        print(f"配置已从 {filepath} 加载")
        return config
    
    def list_configs(self):
        """列出所有可用配置"""
        configs = [f for f in os.listdir(self.config_dir) if f.endswith('.json')]
        return configs

# 预定义的实验配置
def get_normal_config() -> ExperimentConfig:
    """正常模式配置"""
    return ExperimentConfig(
        name="normal_experiment",
        description="正常模式 - 无拓扑变化",
        topology=TopologyConfig(experiment_type='normal')
    )

def get_probabilistic_config() -> ExperimentConfig:
    """概率驱动拓扑变化配置"""
    return ExperimentConfig(
        name="probabilistic_experiment",
        description="概率驱动拓扑变化 - 80%正常/15%损失/5%添加",
        topology=TopologyConfig(
            experiment_type='probabilistic',
            min_active_agents=3,
            max_active_agents=8
        )
    )



# 配置工厂函数
def create_config(config_name: str) -> ExperimentConfig:
    """根据名称创建预定义配置"""
    config_map = {
        'normal': get_normal_config,
        'probabilistic': get_probabilistic_config
    }
    
    if config_name in config_map:
        return config_map[config_name]()
    else:
        raise ValueError(f"未知配置名称: {config_name}. 可用配置: {list(config_map.keys())}")

# 默认配置管理器实例
config_manager = ConfigManager()
