# -*- coding: utf-8 -*-
"""
验证简化环境训练代码的快速测试
"""

import sys
import os

# 获取当前文件的父目录（mpe_uav目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # mpe_uav目录
sys.path.append(parent_dir)

import numpy as np
import torch

# 导入简化环境和MATD3
import importlib.util
uav_env_path = os.path.join(parent_dir, 'uav_top_env', 'uav_env_clean.py')
spec = importlib.util.spec_from_file_location("uav_env_clean", uav_env_path)
uav_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uav_env_module)
UAVEnv = uav_env_module.UAVEnv

from matd3_no_gat import MATD3, ReplayBuffer
from config import CONFIG

# 获取当前文件目录路径
output_dir = os.path.join(current_dir, './output_clean_env')  # 输出目录

def verify_environment():
    """验证环境创建和基本功能"""
    print("🔍 验证环境创建...")
    
    env = UAVEnv(
        render_mode=None,
        experiment_type='probabilistic',
        num_agents=6,
        num_targets=10,
        max_steps=50,  # 缩短用于快速测试
        min_active_agents=3,
        max_active_agents=6
    )
    
    print(f"✅ 环境创建成功: {env.__class__.__name__}")
    print(f"实验类型: {env.experiment_type}")
    
    # 测试重置
    obs, _ = env.reset()
    print(f"✅ 环境重置成功，观察维度: {obs['agent_0'].shape}")
    
    # 测试步进
    actions = {agent: env.get_action_space(agent).sample() for agent in env.agents}
    obs, rewards, dones, _, _ = env.step(actions)
    print(f"✅ 环境步进成功，奖励范围: {min(rewards.values()):.3f} ~ {max(rewards.values()):.3f}")
    
    env.close()
    return True

def verify_matd3():
    """验证MATD3算法"""
    print("\n🔍 验证MATD3算法...")
    
    # 创建环境
    env = UAVEnv(experiment_type='normal', num_agents=4, num_targets=6, max_steps=30)
    obs, _ = env.reset()
    agents = env.agents
    
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]
    
    # 创建MATD3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    
    matd3 = MATD3(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=device,
        actor_lr=1e-4,
        critic_lr=1e-3
    )
    
    print(f"✅ MATD3创建成功，设备: {device}")
    
    # 测试动作选择
    actions = matd3.select_action(obs, noise=0.1)
    print(f"✅ 动作选择成功，动作数量: {len(actions)}")
    
    # 测试经验回放
    replay_buffer = ReplayBuffer(buffer_size=1000, batch_size=32, device=device)
    
    # 收集一些经验
    for step in range(50):
        actions = matd3.select_action(obs, noise=0.1)
        next_obs, rewards, dones, _, _ = env.step(actions)
        
        # 填充所有智能体的经验
        all_agents = set(agents)
        obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
        rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
        next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        dones_filled = {agent: dones.get(agent, True) for agent in all_agents}
        
        replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
        obs = next_obs
    
    print(f"✅ 经验收集成功，缓冲区大小: {len(replay_buffer)}")
    
    # 测试训练
    if len(replay_buffer) > 32:
        loss_info = matd3.train(replay_buffer)
        print(f"✅ 训练成功，Actor损失: {loss_info['actor_loss']:.4f}, Critic损失: {loss_info['critic_loss']:.4f}")
    
    env.close()
    return True

def verify_topology_experiments():
    """验证拓扑实验功能"""
    print("\n🔍 验证拓扑实验功能...")
    
    env = UAVEnv(
        experiment_type='probabilistic',
        num_agents=5,
        num_targets=8,
        max_steps=30
    )
    
    episode_types = []
    topology_changes = []
    
    # 测试多个episodes
    for episode in range(10):
        obs, _ = env.reset()
        episode_type = env.episode_plan['type']
        episode_types.append(episode_type)
        
        initial_uavs = len(env.active_agents)
        
        # 运行episode
        for step in range(env.max_steps):
            actions = {agent: env.get_action_space(agent).sample() for agent in env.agents}
            obs, rewards, dones, _, _ = env.step(actions)
        
        final_uavs = len(env.active_agents)
        topology_changes.append(initial_uavs != final_uavs)
    
    # 统计结果
    normal_count = episode_types.count('normal')
    loss_count = episode_types.count('loss')
    addition_count = episode_types.count('addition')
    change_count = sum(topology_changes)
    
    print(f"✅ 拓扑实验测试完成")
    print(f"Episode类型分布: 正常={normal_count}, 损失={loss_count}, 添加={addition_count}")
    print(f"实际拓扑变化: {change_count}/{loss_count + addition_count}")
    
    env.close()
    return True

def verify_config():
    """验证配置"""
    print("\n🔍 验证配置...")
    
    required_keys = [
        "num_episodes", "max_steps", "initial_random_steps",
        "actor_lr", "critic_lr", "buffer_size", "batch_size",
        "noise_std", "noise_decay", "min_noise"
    ]
    
    missing_keys = [key for key in required_keys if key not in CONFIG]
    
    if missing_keys:
        print(f"❌ 配置缺少键: {missing_keys}")
        return False
    
    print(f"✅ 配置验证成功")
    print(f"训练episodes: {CONFIG['num_episodes']}")
    print(f"最大步数: {CONFIG['max_steps']}")
    print(f"批次大小: {CONFIG['batch_size']}")
    
    return True

def main():
    """主验证函数"""
    print("🚀 开始验证简化环境训练代码\n")
    
    tests = [
        ("环境功能", verify_environment),
        ("MATD3算法", verify_matd3),
        ("拓扑实验", verify_topology_experiments),
        ("配置文件", verify_config)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ 通过" if result else "❌ 失败"
        except Exception as e:
            results[test_name] = f"❌ 错误: {str(e)}"
    
    print("\n" + "="*50)
    print("📊 验证结果总结:")
    print("="*50)
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    passed_tests = sum(1 for result in results.values() if "✅" in result)
    total_tests = len(results)
    
    print(f"\n通过验证: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有验证通过！可以开始训练了。")
        print("\n📝 使用方法:")
        print("1. 运行训练: python main_clean_env.py")
        print("2. 运行测试: python test_clean_env.py")
        print("\n💡 特点:")
        print("- 使用简化环境 (uav_env_clean.py)")
        print("- 支持概率驱动拓扑变化")
        print("- 完整的训练和测试流程")
        print("- 兼容原有MATD3算法")
    else:
        print("\n⚠️  部分验证失败，请检查代码。")

if __name__ == '__main__':
    main()
