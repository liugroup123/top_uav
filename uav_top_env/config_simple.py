"""
简化的UAV拓扑环境配置
保留核心功能，删除复杂的配置系统
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SimpleUAVConfig:
    """简化的UAV环境配置"""
    # 环境基础参数
    num_agents: int = 6
    num_targets: int = 10
    world_size: float = 1.0
    max_steps: int = 200
    
    # 拓扑实验参数
    experiment_type: str = 'normal'  # 'normal' 或 'probabilistic'
    min_active_agents: int = 3
    max_active_agents: Optional[int] = None
    
    # 物理参数
    max_speed: float = 0.1
    communication_range: float = 0.3
    sensing_range: float = 0.2
    
    # 奖励权重（简化）
    coverage_weight: float = 1.0
    connectivity_weight: float = 0.5
    boundary_penalty: float = 0.1
    
    # 概率设置（仅用于probabilistic模式）
    normal_probability: float = 0.80
    loss_probability: float = 0.15
    addition_probability: float = 0.05

# 预定义配置
def get_normal_config() -> SimpleUAVConfig:
    """正常模式配置"""
    return SimpleUAVConfig(experiment_type='normal')

def get_probabilistic_config() -> SimpleUAVConfig:
    """概率驱动拓扑变化配置"""
    return SimpleUAVConfig(
        experiment_type='probabilistic',
        num_agents=8,
        num_targets=12,
        min_active_agents=3,
        max_active_agents=8
    )

# 配置工厂
def create_config(config_name: str) -> SimpleUAVConfig:
    """根据名称创建配置"""
    configs = {
        'normal': get_normal_config,
        'probabilistic': get_probabilistic_config
    }
    
    if config_name not in configs:
        raise ValueError(f"未知配置: {config_name}. 可用配置: {list(configs.keys())}")
    
    return configs[config_name]()
