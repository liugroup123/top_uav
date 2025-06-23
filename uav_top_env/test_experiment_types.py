#!/usr/bin/env python3
"""
测试不同实验类型的UAV拓扑环境
"""

import numpy as np
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env_top import UAVEnv

def test_experiment_type(experiment_type, total_steps=300, render_mode='human'):
    """测试特定实验类型"""
    print(f"\n=== 测试实验类型: {experiment_type} ===")
    
    # 初始化环境
    env = UAVEnv(
        num_agents=6,
        num_targets=12,
        render_mode=render_mode,
        world_size=1.0,
        coverage_radius=0.3,
        communication_radius=0.6,
        experiment_type=experiment_type
    )
    
    # 设置为评估模式
    env.eval()
    
    obs, _ = env.reset()
    
    print(f"初始活跃UAV数量: {len(env.active_agents)}")
    print(f"实验配置: {env.get_experiment_info()}")
    
    topology_changes = []  # 记录拓扑变化
    
    for step in range(total_steps):
        # 生成随机动作
        actions = {
            f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
            for i in range(env.num_agents)
        }
        
        # 记录变化前的状态
        prev_active_count = len(env.active_agents)
        prev_topology_state = env.topology_change['in_progress']
        
        # 执行步进
        obs, rewards, dones, _, _ = env.step(actions)
        
        # 检测拓扑变化
        current_active_count = len(env.active_agents)
        current_topology_state = env.topology_change['in_progress']
        
        if current_active_count != prev_active_count or (current_topology_state and not prev_topology_state):
            change_info = {
                'step': step,
                'type': env.topology_change['change_type'],
                'affected_agent': env.topology_change['affected_agent'],
                'active_count_before': prev_active_count,
                'active_count_after': current_active_count
            }
            topology_changes.append(change_info)
            print(f"Step {step}: 拓扑变化 - {change_info}")
        
        # 渲染
        if render_mode == 'human':
            env.render()
            time.sleep(0.05)  # 稍微快一点
            
        # 每50步打印一次状态
        if step % 50 == 0:
            coverage_rate, connected, max_coverage, unconnected = env.calculate_coverage_complete()
            print(f"Step {step}: 覆盖率={coverage_rate:.2f}, 连通性={connected}, "
                  f"活跃UAV={len(env.active_agents)}")
    
    # 总结
    print(f"\n实验类型 {experiment_type} 完成:")
    print(f"总拓扑变化次数: {len(topology_changes)}")
    print(f"最终活跃UAV数量: {len(env.active_agents)}")
    
    # 统计变化类型
    failure_count = sum(1 for change in topology_changes if change['type'] == 'failure')
    addition_count = sum(1 for change in topology_changes if change['type'] == 'addition')
    print(f"UAV失效次数: {failure_count}")
    print(f"UAV添加次数: {addition_count}")
    
    env.close()
    return topology_changes

def test_dynamic_experiment_switching():
    """测试动态切换实验类型"""
    print(f"\n=== 测试动态实验类型切换 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=10,
        render_mode='human',
        experiment_type='normal'
    )
    
    obs, _ = env.reset()
    
    # 测试序列：normal -> uav_loss -> uav_addition -> random_mixed
    experiment_sequence = [
        ('normal', 100),
        ('uav_loss', 100), 
        ('uav_addition', 100),
        ('random_mixed', 100)
    ]
    
    total_step = 0
    
    for exp_type, duration in experiment_sequence:
        print(f"\n切换到实验类型: {exp_type}")
        env.set_experiment_type(exp_type)
        
        for step in range(duration):
            actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                      for i in range(env.num_agents)}
            
            obs, rewards, dones, _, _ = env.step(actions)
            
            env.render()
            time.sleep(0.05)
            
            total_step += 1
            
            if step % 25 == 0:
                coverage_rate, connected, _, _ = env.calculate_coverage_complete()
                print(f"Step {total_step}: 覆盖率={coverage_rate:.2f}, "
                      f"活跃UAV={len(env.active_agents)}")
    
    env.close()

def compare_experiment_types():
    """比较不同实验类型的性能"""
    print(f"\n=== 比较不同实验类型 ===")
    
    experiment_types = ['normal', 'uav_loss', 'uav_addition', 'random_mixed']
    results = {}
    
    for exp_type in experiment_types:
        print(f"\n运行实验类型: {exp_type}")
        
        env = UAVEnv(
            num_agents=5,
            num_targets=10,
            render_mode=None,  # 不渲染以加快速度
            experiment_type=exp_type
        )
        
        obs, _ = env.reset()
        
        coverage_rates = []
        connectivity_rates = []
        active_agent_counts = []
        
        for step in range(200):
            actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                      for i in range(env.num_agents)}
            
            obs, rewards, dones, _, _ = env.step(actions)
            
            coverage_rate, connected, _, unconnected = env.calculate_coverage_complete()
            coverage_rates.append(coverage_rate)
            connectivity_rates.append(1.0 if connected else 0.0)
            active_agent_counts.append(len(env.active_agents))
        
        results[exp_type] = {
            'avg_coverage': np.mean(coverage_rates),
            'avg_connectivity': np.mean(connectivity_rates),
            'avg_active_agents': np.mean(active_agent_counts),
            'final_active_agents': len(env.active_agents)
        }
        
        env.close()
    
    # 打印比较结果
    print(f"\n=== 实验类型性能比较 ===")
    print(f"{'类型':<15} {'平均覆盖率':<12} {'平均连通率':<12} {'平均活跃UAV':<12} {'最终活跃UAV':<12}")
    print("-" * 65)
    
    for exp_type, result in results.items():
        print(f"{exp_type:<15} {result['avg_coverage']:<12.3f} "
              f"{result['avg_connectivity']:<12.3f} {result['avg_active_agents']:<12.1f} "
              f"{result['final_active_agents']:<12}")

if __name__ == '__main__':
    # 测试各种实验类型
    print("开始测试不同的实验类型...")
    
    # 1. 测试正常模式（无拓扑变化）
    test_experiment_type('normal', total_steps=200, render_mode='human')
    
    # 2. 测试UAV损失模式
    test_experiment_type('uav_loss', total_steps=250, render_mode='human')
    
    # 3. 测试UAV添加模式
    test_experiment_type('uav_addition', total_steps=250, render_mode='human')
    
    # 4. 测试随机混合模式
    test_experiment_type('random_mixed', total_steps=300, render_mode='human')
    
    # 5. 测试动态切换
    # test_dynamic_experiment_switching()
    
    # 6. 性能比较
    # compare_experiment_types()
    
    print("\n所有测试完成！")
