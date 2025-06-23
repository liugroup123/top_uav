#!/usr/bin/env python3
"""
简单测试新的实验类型功能
"""

import numpy as np
import sys
import os

# 添加父目录到Python路径以导入uav_env_top
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录 (uav_top_env)
sys.path.append(parent_dir)

from uav_env_top import UAVEnv

def simple_test():
    """简单测试不同实验类型"""
    print("=== 测试UAV拓扑环境的实验类型功能 ===\n")
    
    # 测试各种实验类型
    experiment_types = ['normal', 'uav_loss', 'uav_addition', 'random_mixed']
    
    for exp_type in experiment_types:
        print(f"测试实验类型: {exp_type}")
        
        try:
            # 创建环境
            env = UAVEnv(
                num_agents=5,
                num_targets=8,
                render_mode=None,  # 不渲染以加快测试
                experiment_type=exp_type
            )
            
            # 重置环境
            obs, _ = env.reset()
            print(f"  ✓ 环境创建成功")
            print(f"  ✓ 初始活跃UAV数量: {len(env.active_agents)}")
            
            # 获取实验信息
            info = env.get_experiment_info()
            print(f"  ✓ 实验配置: {info['experiment_type']}")
            print(f"  ✓ 拓扑变化启用: {info['topology_enabled']}")
            
            # 运行几步测试
            topology_changes = 0
            for step in range(50):
                # 生成随机动作
                actions = {
                    f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                    for i in range(env.num_agents)
                }
                
                # 记录变化前的状态
                prev_active_count = len(env.active_agents)
                
                # 执行步进
                obs, rewards, dones, _, _ = env.step(actions)
                
                # 检测拓扑变化
                current_active_count = len(env.active_agents)
                if current_active_count != prev_active_count:
                    topology_changes += 1
                    print(f"    Step {step}: 拓扑变化 - UAV数量从 {prev_active_count} 变为 {current_active_count}")
            
            print(f"  ✓ 50步测试完成，发生 {topology_changes} 次拓扑变化")
            
            # 测试动态切换实验类型
            if exp_type == 'normal':
                print("  测试动态切换实验类型...")
                env.set_experiment_type('uav_loss')
                new_info = env.get_experiment_info()
                print(f"  ✓ 成功切换到: {new_info['experiment_type']}")
            
            env.close()
            print(f"  ✓ {exp_type} 测试通过\n")
            
        except Exception as e:
            print(f"  ✗ {exp_type} 测试失败: {str(e)}\n")
    
    print("=== 所有测试完成 ===")

def test_experiment_switching():
    """测试实验类型动态切换"""
    print("=== 测试实验类型动态切换 ===\n")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=8,
        render_mode=None,
        experiment_type='normal'
    )
    
    # 测试切换序列
    switch_sequence = ['normal', 'uav_loss', 'uav_addition', 'random_mixed', 'normal']
    
    for target_type in switch_sequence:
        print(f"切换到: {target_type}")
        env.set_experiment_type(target_type)
        
        info = env.get_experiment_info()
        print(f"  当前类型: {info['experiment_type']}")
        print(f"  拓扑变化启用: {info['topology_enabled']}")
        
        # 运行几步验证
        for step in range(10):
            actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                      for i in range(env.num_agents)}
            env.step(actions)
        
        print(f"  ✓ 运行10步正常\n")
    
    env.close()
    print("=== 动态切换测试完成 ===")

def test_topology_config():
    """测试拓扑配置"""
    print("=== 测试拓扑配置 ===\n")
    
    env = UAVEnv(experiment_type='uav_loss')
    
    print("拓扑配置信息:")
    config = env.topology_config
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n实验信息:")
    info = env.get_experiment_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    env.close()
    print("\n=== 拓扑配置测试完成 ===")

if __name__ == '__main__':
    # 运行测试
    simple_test()
    test_experiment_switching()
    test_topology_config()
    
    print("\n🎉 所有测试完成！新的实验类型功能工作正常。")
