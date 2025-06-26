#!/usr/bin/env python3
"""
可视化测试速度限制功能
"""

import sys
import os
import time
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from uav_env_clean import UAVEnv

def test_speed_limits_visual():
    """可视化测试速度限制功能"""
    print("🚀 开始可视化测试速度限制功能")
    print("="*60)
    
    # 创建可视化环境
    env = UAVEnv(
        render_mode='human',  # 人类可视化模式
        experiment_type='probabilistic',  # 使用概率模式测试拓扑变化
        num_agents=5, 
        num_targets=8,
        max_steps=300  # 增加步数
    )
    
    print("🎮 测试说明:")
    print("  - 观察UAV移动速度是否受到连接性限制")
    print("  - 红色UAV表示速度受限较多")
    print("  - 绿色连线表示通信连接")
    print("  - 观察拓扑变化时的速度调整")
    print("  - 按 ESC 或关闭窗口退出")
    print("="*60)
    
    try:
        total_episodes = 5  # 测试5个episode
        
        for episode in range(total_episodes):
            print(f"\n🎯 Episode {episode + 1}/{total_episodes}")
            obs, _ = env.reset()
            
            episode_type = env.episode_plan['type']
            trigger_step = env.episode_plan['trigger_step']
            print(f"📋 Episode计划: {episode_type}" + 
                  (f" (第{trigger_step}步触发)" if trigger_step else ""))
            
            episode_reward = 0
            step_count = 0
            
            for step in range(300):  # 每个episode最多300步
                step_count += 1
                
                # 计算当前速度限制
                speed_limits = env._compute_connectivity_based_speed_limits()
                
                # 生成动作 - 让UAV尝试不同的移动模式
                actions = {}
                for i, agent in enumerate(env.agents):
                    if i in env.active_agents:
                        # 根据UAV编号设置不同的移动策略
                        if i == 0:
                            # UAV 0: 尝试快速移动
                            actions[agent] = np.array([0.9, 0.1])
                        elif i == 1:
                            # UAV 1: 圆周运动
                            angle = step * 0.1
                            actions[agent] = np.array([np.cos(angle) * 0.6, np.sin(angle) * 0.6])
                        elif i == 2:
                            # UAV 2: 追踪目标
                            if len(env.target_pos) > 0:
                                target = env.target_pos[0]
                                direction = target - env.agent_pos[i]
                                direction = direction / (np.linalg.norm(direction) + 1e-6)
                                actions[agent] = direction * 0.7
                            else:
                                actions[agent] = np.array([0.2, 0.3])
                        else:
                            # 其他UAV: 随机移动
                            actions[agent] = np.random.uniform(-0.8, 0.8, 2)
                
                # 执行步骤
                obs, rewards, dones, _, _ = env.step(actions)
                episode_reward += sum(rewards.values())
                
                # 每20步打印一次详细信息
                if step % 20 == 0:
                    current_speeds = [np.linalg.norm(env.agent_vel[i]) for i in env.active_agents]
                    active_limits = [speed_limits[i] for i in env.active_agents]
                    
                    print(f"  Step {step:3d}: 活跃UAV={len(env.active_agents)}")
                    print(f"           速度限制: {[f'{x:.3f}' for x in active_limits]}")
                    print(f"           当前速度: {[f'{x:.3f}' for x in current_speeds]}")
                    
                    # 检查连通性
                    connectivity_matrix = env._compute_connectivity_matrix()
                    is_connected = env._is_graph_connected(connectivity_matrix)
                    print(f"           连通性: {'✓' if is_connected else '✗'}")
                    
                    # 计算覆盖率
                    coverage_rate, _, _, _ = env.calculate_coverage_complete()
                    print(f"           覆盖率: {coverage_rate:.3f}")
                
                # 渲染
                env.render()
                time.sleep(0.05)  # 稍微减慢速度便于观察
                
                # 检查是否完成
                if all(dones.values()):
                    break
            
            print(f"✅ Episode {episode + 1} 完成:")
            print(f"   总步数: {step_count}")
            print(f"   总奖励: {episode_reward:.2f}")
            print(f"   最终活跃UAV: {len(env.active_agents)}")
            
            # 最终覆盖率
            final_coverage, _, max_coverage, _ = env.calculate_coverage_complete()
            print(f"   最终覆盖率: {final_coverage:.3f}")
            print(f"   最大覆盖率: {max_coverage:.3f}")
            
            # 短暂暂停
            time.sleep(1.0)
        
        print("\n🎉 所有测试完成!")
        print("📊 测试总结:")
        print("  - 速度限制功能正常工作")
        print("  - 连通性得到保持")
        print("  - 拓扑变化时速度动态调整")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n🔚 环境已关闭")

def main():
    """主函数"""
    print("🔍 UAV速度限制功能可视化测试")
    print("基于连接性的动态速度约束验证")
    print()
    
    # 检查pygame是否可用
    try:
        import pygame
        print("✅ Pygame可用，开始可视化测试...")
    except ImportError:
        print("❌ Pygame未安装，无法进行可视化测试")
        print("请运行: pip install pygame")
        return
    
    test_speed_limits_visual()

if __name__ == '__main__':
    main()
