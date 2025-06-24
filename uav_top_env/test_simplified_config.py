#!/usr/bin/env python3
"""
测试简化后的配置文件
"""

from config import (
    get_normal_config, 
    get_uav_loss_config, 
    get_uav_addition_config,
    create_config
)

def test_simplified_configs():
    """测试三种核心配置"""
    print("🔍 测试简化后的配置文件...\n")
    
    # 测试三种核心模式
    configs = [
        ("正常模式", get_normal_config()),
        ("UAV损失模式", get_uav_loss_config()),
        ("UAV添加模式", get_uav_addition_config())
    ]
    
    for mode_name, config in configs:
        print(f"=== {mode_name} ===")
        print(f"配置名称: {config.name}")
        print(f"描述: {config.description}")
        print(f"实验类型: {config.topology.experiment_type}")
        
        if config.topology.topology_change_interval:
            print(f"变化间隔: {config.topology.topology_change_interval}步")
        if config.topology.min_active_agents:
            print(f"最少UAV: {config.topology.min_active_agents}")
        if config.topology.initial_active_ratio:
            print(f"初始活跃比例: {config.topology.initial_active_ratio}")
        
        print(f"环境配置: {config.environment.num_agents}个UAV, {config.environment.num_targets}个目标")
        print()
    
    # 测试工厂函数
    print("=== 测试配置工厂函数 ===")
    try:
        for config_name in ['normal', 'uav_loss', 'uav_addition']:
            config = create_config(config_name)
            print(f"✅ {config_name}: {config.description}")
        
        # 测试错误配置
        try:
            create_config('invalid_config')
        except ValueError as e:
            print(f"✅ 错误处理正常: {e}")
            
    except Exception as e:
        print(f"❌ 配置工厂测试失败: {e}")
    
    print("\n🎉 配置简化完成！现在只有三种核心模式：")
    print("1. normal - 正常模式（无拓扑变化）")
    print("2. uav_loss - UAV损失模式（UAV失效）") 
    print("3. uav_addition - UAV添加模式（添加新UAV）")

if __name__ == '__main__':
    test_simplified_configs()
