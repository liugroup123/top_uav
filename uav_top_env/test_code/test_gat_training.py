#!/usr/bin/env python3
"""
测试GAT网络是否参与训练的示例脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# 添加父目录到Python路径以导入uav_env_top
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录 (uav_top_env)
sys.path.append(parent_dir)

from uav_env_top import UAVEnv

def test_gat_parameters():
    """测试GAT参数是否可访问"""
    print("=== 测试GAT参数访问 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    # 获取GAT信息
    gat_info = env.get_gat_info()
    print(f"GAT总参数数量: {gat_info['total_parameters']}")
    print(f"GAT可训练参数数量: {gat_info['trainable_parameters']}")
    print(f"GAT设备: {gat_info['device']}")
    print(f"GAT训练模式: {gat_info['training_mode']}")
    
    # 检查参数是否可访问
    param_count = 0
    for name, param in env.get_gat_named_parameters():
        param_count += param.numel()
        print(f"参数: {name}, 形状: {param.shape}, 需要梯度: {param.requires_grad}")
    
    print(f"通过named_parameters访问的参数总数: {param_count}")
    
    env.close()
    return gat_info['trainable_parameters'] > 0

def test_gat_gradient_flow():
    """测试GAT梯度流"""
    print("\n=== 测试GAT梯度流 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    env.reset()
    
    # 获取带梯度的GAT特征
    print("获取带梯度的GAT特征...")
    gat_features = env.get_gat_features_with_grad()
    
    print(f"GAT特征形状: {gat_features.shape}")
    print(f"GAT特征需要梯度: {gat_features.requires_grad}")
    
    # 创建一个简单的损失
    target = torch.zeros_like(gat_features)
    loss = nn.MSELoss()(gat_features, target)
    print(f"损失值: {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_count = 0
    for name, param in env.get_gat_named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm = param.grad.norm().item()
            print(f"参数 {name} 梯度范数: {grad_norm:.6f}")
    
    print(f"有梯度的参数数量: {grad_count}")
    
    env.close()
    return grad_count > 0

def test_gat_training_loop():
    """测试GAT训练循环"""
    print("\n=== 测试GAT训练循环 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    # 创建GAT优化器
    gat_optimizer = optim.Adam(env.get_gat_parameters(), lr=0.001)
    
    env.reset()
    
    print("开始训练循环...")
    initial_loss = None
    final_loss = None
    
    for epoch in range(10):
        # 获取带梯度的GAT特征
        gat_features = env.get_gat_features_with_grad()
        
        # 创建一个目标（这里只是示例，实际应该来自RL算法）
        target = torch.randn_like(gat_features) * 0.1
        
        # 计算损失
        loss = nn.MSELoss()(gat_features, target)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        # 更新GAT参数
        env.update_gat_with_loss(loss, gat_optimizer)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        final_loss = loss.item()
    
    print(f"初始损失: {initial_loss:.6f}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"损失变化: {initial_loss - final_loss:.6f}")
    
    env.close()
    return abs(initial_loss - final_loss) > 1e-6

def test_gat_save_load():
    """测试GAT模型保存和加载"""
    print("\n=== 测试GAT模型保存和加载 ===")
    
    env1 = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    # 获取初始参数
    initial_state = env1.get_gat_state_dict()
    
    # 保存模型
    model_path = "test_gat_model.pth"
    env1.save_gat_model(model_path)
    
    # 修改参数（模拟训练）
    for param in env1.get_gat_parameters():
        param.data += torch.randn_like(param.data) * 0.01
    
    # 创建新环境并加载模型
    env2 = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    env2.load_gat_model(model_path)
    loaded_state = env2.get_gat_state_dict()
    
    # 比较参数
    params_match = True
    for key in initial_state.keys():
        if not torch.allclose(initial_state[key], loaded_state[key]):
            params_match = False
            break
    
    print(f"参数匹配: {params_match}")
    
    # 清理
    if os.path.exists(model_path):
        os.remove(model_path)
    
    env1.close()
    env2.close()
    return params_match

def test_gat_in_environment_step():
    """测试GAT在环境step中的使用"""
    print("\n=== 测试GAT在环境step中的使用 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=8,
        experiment_type='normal',
        render_mode=None
    )
    
    obs, _ = env.reset()
    
    # 检查观察中是否包含GAT特征
    print(f"观察空间维度: {env.observation_space[0].shape}")
    print(f"实际观察维度: {obs['agent_0'].shape}")
    
    # 运行几步
    for step in range(5):
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
                  for i in range(env.num_agents)}
        
        obs, rewards, dones, _, _ = env.step(actions)
        
        # 检查GAT特征是否在观察中
        agent_0_obs = obs['agent_0']
        gat_features_start = 4 + 2*env.num_targets + 2*(env.num_agents-1)
        gat_features_end = gat_features_start + 32  # GAT特征维度
        
        gat_part = agent_0_obs[gat_features_start:gat_features_end]
        gat_norm = np.linalg.norm(gat_part)
        
        print(f"Step {step}: GAT特征范数 = {gat_norm:.6f}")
    
    env.close()
    return True

def main():
    """主测试函数"""
    print("🔍 开始测试GAT是否参与训练...\n")
    
    tests = [
        ("GAT参数访问", test_gat_parameters),
        ("GAT梯度流", test_gat_gradient_flow),
        ("GAT训练循环", test_gat_training_loop),
        ("GAT保存加载", test_gat_save_load),
        ("GAT在环境中使用", test_gat_in_environment_step)
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
        print("\n🎉 所有测试通过！GAT网络可以正常参与训练。")
    else:
        print("\n⚠️  部分测试失败，请检查GAT配置。")

if __name__ == '__main__':
    main()
