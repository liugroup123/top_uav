#!/usr/bin/env python3
"""
测试MATD3是否能正常训练拓扑UAV环境
"""

import sys
import os
import torch
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入模块
from uav_top_env.uav_env_top import UAVEnv
from matd3_no_gat import MATD3, ReplayBuffer
from config import CONFIG

def test_environment_compatibility():
    """测试环境兼容性"""
    print("=== 测试环境兼容性 ===")
    
    try:
        # 测试不同实验类型
        experiment_types = ['normal', 'uav_loss', 'uav_addition', 'random_mixed']
        
        for exp_type in experiment_types:
            print(f"\n测试实验类型: {exp_type}")
            
            # 创建环境
            env = UAVEnv(
                render_mode=None,
                experiment_type=exp_type,
                num_agents=6,
                num_targets=10
            )
            
            # 重置环境
            obs, _ = env.reset()
            agents = env.agents
            
            print(f"  ✓ 环境创建成功")
            print(f"  ✓ 智能体数量: {len(agents)}")
            print(f"  ✓ 观察空间维度: {env.get_observation_space(agents[0]).shape}")
            print(f"  ✓ 动作空间维度: {env.get_action_space(agents[0]).shape}")
            print(f"  ✓ 初始活跃UAV: {len(env.active_agents)}")
            
            # 运行几步
            for step in range(5):
                actions = {agent: env.get_action_space(agent).sample() for agent in agents}
                obs, rewards, dones, _, _ = env.step(actions)
                
                if step == 0:
                    print(f"  ✓ 第一步执行成功")
                    print(f"  ✓ 奖励数量: {len(rewards)}")
                    print(f"  ✓ 观察数量: {len(obs)}")
            
            print(f"  ✓ 最终活跃UAV: {len(env.active_agents)}")
            env.close()
            
        return True
        
    except Exception as e:
        print(f"  ✗ 环境测试失败: {str(e)}")
        return False

def test_matd3_compatibility():
    """测试MATD3兼容性"""
    print("\n=== 测试MATD3兼容性 ===")
    
    try:
        # 创建环境
        env = UAVEnv(
            render_mode=None,
            experiment_type='uav_loss',
            num_agents=5,
            num_targets=8
        )
        
        obs, _ = env.reset()
        agents = env.agents
        
        # 获取观测和动作空间维度
        obs_dim = env.get_observation_space(agents[0]).shape[0]
        action_dim = env.get_action_space(agents[0]).shape[0]
        
        print(f"  观察维度: {obs_dim}")
        print(f"  动作维度: {action_dim}")
        
        # 创建观测和动作维度的字典
        obs_dims = {agent: obs_dim for agent in agents}
        action_dims = {agent: action_dim for agent in agents}
        
        # 初始化 MATD3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matd3 = MATD3(
            agents=agents,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=device,
            actor_lr=1e-4,
            critic_lr=1e-3
        )
        
        print(f"  ✓ MATD3创建成功")
        
        # 测试动作选择
        actions = matd3.select_action(obs, noise=0.1)
        print(f"  ✓ 动作选择成功，动作数量: {len(actions)}")
        
        # 测试环境步进
        next_obs, rewards, dones, _, _ = env.step(actions)
        print(f"  ✓ 环境步进成功")
        
        # 测试经验回放
        replay_buffer = ReplayBuffer(
            buffer_size=1000,
            batch_size=32,
            device=device
        )
        
        # 添加经验
        all_agents = set(agents)
        obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
        rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
        next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        dones_filled = {agent: dones.get(agent, True) for agent in all_agents}
        
        replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
        print(f"  ✓ 经验添加成功")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ MATD3测试失败: {str(e)}")
        return False

def test_topology_changes():
    """测试拓扑变化"""
    print("\n=== 测试拓扑变化 ===")
    
    try:
        env = UAVEnv(
            render_mode=None,
            experiment_type='random_mixed',
            num_agents=6,
            num_targets=8,
            topology_change_probability=0.1  # 高概率便于测试
        )
        
        obs, _ = env.reset()
        initial_active = len(env.active_agents)
        print(f"  初始活跃UAV: {initial_active}")
        
        topology_changes = 0
        for step in range(50):
            actions = {agent: env.get_action_space(agent).sample() for agent in env.agents}
            
            prev_active = len(env.active_agents)
            obs, rewards, dones, _, _ = env.step(actions)
            current_active = len(env.active_agents)
            
            if current_active != prev_active:
                topology_changes += 1
                change_type = "失效" if current_active < prev_active else "添加"
                print(f"    Step {step}: {change_type} - UAV数量 {prev_active} -> {current_active}")
        
        print(f"  ✓ 50步内发生 {topology_changes} 次拓扑变化")
        print(f"  ✓ 最终活跃UAV: {len(env.active_agents)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ 拓扑变化测试失败: {str(e)}")
        return False

def test_training_loop():
    """测试简化的训练循环"""
    print("\n=== 测试训练循环 ===")
    
    try:
        env = UAVEnv(
            render_mode=None,
            experiment_type='uav_loss',
            num_agents=5,
            num_targets=8
        )
        
        obs, _ = env.reset()
        agents = env.agents
        
        # 获取维度
        obs_dim = env.get_observation_space(agents[0]).shape[0]
        action_dim = env.get_action_space(agents[0]).shape[0]
        obs_dims = {agent: obs_dim for agent in agents}
        action_dims = {agent: action_dim for agent in agents}
        
        # 创建MATD3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matd3 = MATD3(
            agents=agents,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=device
        )
        
        # 创建经验回放
        replay_buffer = ReplayBuffer(buffer_size=1000, batch_size=32, device=device)
        
        # 简化训练循环
        total_reward = 0
        for episode in range(3):  # 只测试3个episode
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(20):  # 每个episode只运行20步
                # 选择动作
                actions = matd3.select_action(obs, noise=0.1)
                
                # 环境步进
                next_obs, rewards, dones, _, _ = env.step(actions)
                episode_reward += sum(rewards.values())
                
                # 添加经验
                all_agents = set(agents)
                obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
                actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
                rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
                next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
                dones_filled = {agent: dones.get(agent, True) for agent in all_agents}
                
                replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
                
                # 训练（如果有足够经验）
                if len(replay_buffer) > 32:
                    loss_info = matd3.train(replay_buffer)
                    if episode == 0 and step == 0:
                        print(f"    首次训练成功，Actor损失: {loss_info['actor_loss']:.4f}")
                
                obs = next_obs
            
            total_reward += episode_reward
            print(f"  Episode {episode + 1}: 奖励 = {episode_reward:.2f}, 活跃UAV = {len(env.active_agents)}")
        
        print(f"  ✓ 训练循环测试完成，平均奖励: {total_reward/3:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ 训练循环测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🔍 开始测试MATD3与拓扑UAV环境的兼容性...\n")
    
    tests = [
        ("环境兼容性", test_environment_compatibility),
        ("MATD3兼容性", test_matd3_compatibility),
        ("拓扑变化", test_topology_changes),
        ("训练循环", test_training_loop)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ 通过" if result else "❌ 失败"
        except Exception as e:
            results[test_name] = f"❌ 错误: {str(e)}"
    
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print("="*50)
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # 总结
    passed_tests = sum(1 for result in results.values() if "✅" in result)
    total_tests = len(results)
    
    print(f"\n通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！MATD3可以正常训练拓扑UAV环境。")
        print("\n📝 建议:")
        print("- 可以直接运行 main_no_gat.py 开始训练")
        print("- 可以在环境初始化时修改 experiment_type 来测试不同的拓扑变化")
        print("- 建议先用 'normal' 模式训练基础策略，再用其他模式训练")
    else:
        print("\n⚠️  部分测试失败，请检查配置。")

if __name__ == '__main__':
    main()
