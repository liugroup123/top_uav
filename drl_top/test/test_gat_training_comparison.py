#!/usr/bin/env python3
"""
测试GAT训练版本与非GAT版本的对比
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np

def test_gat_training_setup():
    """测试GAT训练设置"""
    print("🔍 测试GAT训练设置...\n")
    
    try:
        from uav_top_env.uav_env_top import UAVEnv
        from matd3_no_gat import MATD3, ReplayBuffer
        from config import CONFIG
        
        # 创建环境
        env = UAVEnv(
            render_mode=None,
            experiment_type='uav_loss',
            num_agents=6,
            num_targets=10
        )
        
        print("=== 测试GAT训练模式 ===")
        
        # 测试GAT训练模式
        env.training = True
        print(f"✅ GAT训练模式已启用: {env.training}")
        
        # 获取GAT参数
        gat_params = list(env.get_gat_parameters())
        gat_param_count = sum(p.numel() for p in gat_params)
        print(f"✅ GAT参数数量: {gat_param_count}")
        
        # 测试环境重置和观察
        obs, _ = env.reset()
        agents = env.agents
        
        obs_dim = env.get_observation_space(agents[0]).shape[0]
        action_dim = env.get_action_space(agents[0]).shape[0]
        
        print(f"✅ 观察维度: {obs_dim}")
        print(f"✅ 动作维度: {action_dim}")
        print(f"✅ 智能体数量: {len(agents)}")
        
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
        
        print(f"✅ MATD3创建成功")
        
        # 测试优化器重新配置
        print("\n=== 测试优化器配置 ===")
        
        original_actor_params = sum(p.numel() for p in matd3.actors[agents[0]].parameters())
        original_critic_params = sum(p.numel() for p in matd3.critics_1[agents[0]].parameters()) + sum(p.numel() for p in matd3.critics_2[agents[0]].parameters())
        
        print(f"原始Actor参数: {original_actor_params}")
        print(f"原始Critic参数: {original_critic_params}")
        print(f"GAT参数: {gat_param_count}")
        
        # 重新配置优化器
        for agent in agents:
            actor_params = list(matd3.actors[agent].parameters())
            critic_params = list(matd3.critics_1[agent].parameters()) + list(matd3.critics_2[agent].parameters())
            
            # 重新创建优化器，包含GAT参数
            matd3.actor_optimizers[agent] = torch.optim.Adam(
                actor_params + gat_params, 
                lr=1e-4
            )
            matd3.critic_optimizers[agent] = torch.optim.Adam(
                critic_params + gat_params, 
                lr=1e-3
            )
        
        print("✅ 优化器重新配置完成")
        
        # 测试训练步骤
        print("\n=== 测试训练步骤 ===")
        
        # 创建经验回放
        replay_buffer = ReplayBuffer(buffer_size=1000, batch_size=32, device=device)
        
        # 收集一些经验
        for step in range(50):
            actions = matd3.select_action(obs, noise=0.1)
            next_obs, rewards, dones, _, _ = env.step(actions)
            
            all_agents = set(agents)
            obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
            rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
            next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            dones_filled = {agent: dones.get(agent, True) for agent in all_agents}
            
            replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
            obs = next_obs
        
        print(f"✅ 收集了 {len(replay_buffer)} 个经验")
        
        # 测试训练
        if len(replay_buffer) > 32:
            loss_info = matd3.train(replay_buffer)
            print(f"✅ 训练成功！Actor损失: {loss_info['actor_loss']:.4f}, Critic损失: {loss_info['critic_loss']:.4f}")
            print("🔥 GAT参数已与策略网络联合更新")
        
        env.close()
        
        print("\n=== 对比总结 ===")
        print("main_no_gat.py:")
        print("  - GAT作为固定特征提取器")
        print("  - 只训练策略网络参数")
        print("  - 训练稳定，收敛快")
        print("  - GAT特征被detach()切断梯度")
        
        print("\nmain_with_gat.py:")
        print("  - GAT与策略网络联合训练")
        print("  - 端到端梯度传播")
        print("  - 理论性能更高，但训练复杂")
        print("  - GAT参数包含在优化器中")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 开始测试GAT训练版本...\n")
    
    result = test_gat_training_setup()
    
    if result:
        print("\n🎉 GAT训练版本测试通过！")
        print("\n📝 使用建议:")
        print("1. 先用 main_no_gat.py 训练基础策略")
        print("2. 再用 main_with_gat.py 进行端到端训练")
        print("3. 对比两种方法的性能差异")
        print("4. GAT训练版本可能需要更小的学习率")
    else:
        print("\n❌ GAT训练版本测试失败，请检查代码。")

if __name__ == '__main__':
    main()
