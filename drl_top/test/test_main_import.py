#!/usr/bin/env python3
"""
测试main_no_gat.py的导入是否正常
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print("当前目录:", current_dir)
print("父目录:", parent_dir)

try:
    print("\n1. 测试导入UAV环境...")
    from uav_top_env.uav_env_top import UAVEnv
    print("✅ UAV环境导入成功")
    
    print("\n2. 测试导入MATD3...")
    from matd3_no_gat import MATD3, ReplayBuffer
    print("✅ MATD3导入成功")
    
    print("\n3. 测试导入配置...")
    from config import CONFIG
    print("✅ 配置导入成功")
    print(f"配置内容: {list(CONFIG.keys())}")
    
    print("\n4. 测试创建环境...")
    env = UAVEnv(
        render_mode=None,
        experiment_type='uav_loss',
        num_agents=6,
        num_targets=10
    )
    print("✅ 环境创建成功")
    
    print("\n5. 测试环境重置...")
    obs, _ = env.reset()
    print("✅ 环境重置成功")
    print(f"观察数量: {len(obs)}")
    print(f"观察维度: {obs['agent_0'].shape}")
    
    env.close()
    print("\n🎉 所有导入测试通过！main_no_gat.py应该可以正常运行。")
    
except Exception as e:
    print(f"❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
