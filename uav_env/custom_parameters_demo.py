#!/usr/bin/env python3
"""
演示如何自定义拓扑变化参数
"""

import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env_top import UAVEnv

def demo_custom_parameters():
    """演示自定义参数的使用"""
    print("=== 自定义拓扑变化参数演示 ===\n")
    
    # 示例1: 自定义UAV损失模式 - 更频繁的失效
    print("1. 自定义UAV损失模式 - 每30步失效一个UAV，最少保持2个")
    env1 = UAVEnv(
        num_agents=6,
        num_targets=8,
        experiment_type='uav_loss',
        topology_change_interval=30,  # 每30步变化一次（默认80步）
        min_active_agents=2,          # 最少保持2个UAV（默认3个）
        render_mode=None
    )
    
    test_environment(env1, "频繁UAV损失", 150)
    
    # 示例2: 自定义UAV添加模式 - 从更少的UAV开始
    print("\n2. 自定义UAV添加模式 - 从50%UAV开始，每40步添加一个")
    env2 = UAVEnv(
        num_agents=6,
        num_targets=8,
        experiment_type='uav_addition',
        topology_change_interval=40,   # 每40步变化一次（默认60步）
        initial_active_ratio=0.5,     # 从50%的UAV开始（默认约67%）
        render_mode=None
    )
    
    test_environment(env2, "自定义UAV添加", 150)
    
    # 示例3: 自定义随机混合模式 - 更高的变化概率
    print("\n3. 自定义随机混合模式 - 更高的变化概率")
    env3 = UAVEnv(
        num_agents=6,
        num_targets=8,
        experiment_type='random_mixed',
        topology_change_probability=0.03,  # 每步3%概率变化（默认1.5%）
        min_active_agents=2,               # 最少2个UAV
        max_active_agents=8,               # 最多8个UAV（超过初始数量）
        render_mode=None
    )
    
    test_environment(env3, "高频随机变化", 200)
    
    # 示例4: 极端参数测试
    print("\n4. 极端参数测试 - 非常频繁的变化")
    env4 = UAVEnv(
        num_agents=8,
        num_targets=10,
        experiment_type='random_mixed',
        topology_change_probability=0.05,  # 每步5%概率变化
        min_active_agents=1,               # 最少1个UAV
        max_active_agents=10,              # 最多10个UAV
        render_mode=None
    )
    
    test_environment(env4, "极端变化频率", 100)

def test_environment(env, description, steps):
    """测试环境并打印统计信息"""
    print(f"\n--- {description} ---")
    
    # 打印配置信息
    config = env.topology_config
    print(f"配置: 间隔={config.get('change_interval', 'N/A')}步, "
          f"概率={config.get('change_probability', 'N/A')}, "
          f"最少UAV={config['min_agents']}, 最多UAV={config['max_agents']}")
    
    obs, _ = env.reset()
    print(f"初始活跃UAV: {len(env.active_agents)}")
    
    topology_changes = []
    uav_counts = [len(env.active_agents)]
    
    for step in range(steps):
        # 生成随机动作
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                  for i in range(env.num_agents)}
        
        prev_count = len(env.active_agents)
        obs, rewards, dones, _, _ = env.step(actions)
        current_count = len(env.active_agents)
        
        uav_counts.append(current_count)
        
        # 记录拓扑变化
        if current_count != prev_count:
            change_type = "失效" if current_count < prev_count else "添加"
            topology_changes.append({
                'step': step,
                'type': change_type,
                'from': prev_count,
                'to': current_count
            })
            print(f"  Step {step}: UAV {change_type} ({prev_count} -> {current_count})")
    
    # 统计信息
    print(f"总变化次数: {len(topology_changes)}")
    print(f"最终UAV数量: {len(env.active_agents)}")
    print(f"UAV数量范围: {min(uav_counts)} - {max(uav_counts)}")
    
    # 变化类型统计
    failures = sum(1 for c in topology_changes if c['type'] == '失效')
    additions = sum(1 for c in topology_changes if c['type'] == '添加')
    print(f"失效次数: {failures}, 添加次数: {additions}")
    
    env.close()

def demo_parameter_comparison():
    """比较不同参数设置的效果"""
    print("\n=== 参数设置效果比较 ===\n")
    
    # 不同间隔设置的比较
    intervals = [20, 50, 100]
    
    for interval in intervals:
        print(f"测试间隔: {interval}步")
        env = UAVEnv(
            num_agents=5,
            experiment_type='uav_loss',
            topology_change_interval=interval,
            render_mode=None
        )
        
        obs, _ = env.reset()
        changes = 0
        
        for step in range(200):
            actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                      for i in range(env.num_agents)}
            prev_count = len(env.active_agents)
            env.step(actions)
            if len(env.active_agents) != prev_count:
                changes += 1
        
        print(f"  200步内发生 {changes} 次变化")
        env.close()

def demo_dynamic_parameter_adjustment():
    """演示动态调整参数"""
    print("\n=== 动态参数调整演示 ===\n")
    
    env = UAVEnv(
        num_agents=6,
        experiment_type='uav_loss',
        topology_change_interval=50,
        render_mode=None
    )
    
    obs, _ = env.reset()
    
    # 运行前100步
    print("前100步: 默认参数")
    for step in range(100):
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                  for i in range(env.num_agents)}
        env.step(actions)
    
    print(f"100步后UAV数量: {len(env.active_agents)}")
    
    # 动态调整参数
    print("调整参数: 更频繁的变化")
    env.topology_config['change_interval'] = 20  # 直接修改配置
    env.topology_config['min_agents'] = 2
    
    # 继续运行100步
    changes = 0
    for step in range(100, 200):
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                  for i in range(env.num_agents)}
        prev_count = len(env.active_agents)
        env.step(actions)
        if len(env.active_agents) != prev_count:
            changes += 1
    
    print(f"调整后100步内发生 {changes} 次变化")
    print(f"最终UAV数量: {len(env.active_agents)}")
    
    env.close()

if __name__ == '__main__':
    # 运行演示
    demo_custom_parameters()
    demo_parameter_comparison()
    demo_dynamic_parameter_adjustment()
    
    print("\n🎉 自定义参数演示完成！")
    print("\n📝 总结:")
    print("- topology_change_interval: 控制固定间隔模式的变化频率")
    print("- topology_change_probability: 控制随机模式的变化概率")
    print("- min_active_agents: 设置最少保持的UAV数量")
    print("- max_active_agents: 设置最多允许的UAV数量")
    print("- initial_active_ratio: 设置UAV添加模式的初始活跃比例")
    print("- 所有参数都可以在运行时动态调整！")
