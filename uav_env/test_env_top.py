import time
import numpy as np
import os,sys
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 修改这里：从 uav_env_top.py 导入环境
from mpe_uav.uav_env.uav_env_top import UAVEnv  # 改用支持拓扑变化的环境
import pdb

def test_topology_changes(render_mode='human'):
    print("\n=== 测试动态拓扑变化 ===")
    
    # 初始化环境
    env = UAVEnv(
        num_agents=6,
        num_targets=10,
        render_mode=render_mode,
        world_size=1.0,
        coverage_radius=0.3,
        communication_radius=0.6
    )
    
    # 设置为评估模式
    env.eval()
    
    obs, _ = env.reset()

    # 测试阶段1：正常运行
    print("\n第一阶段：系统正常运行")
    for step in range(50):  # 先让系统运行一段时间达到稳定
        # 确保动作是float32类型
        actions = {
            f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
            for i in range(env.num_agents)
        }
        obs, rewards, dones, _, _ = env.step(actions)
        
        if render_mode == 'human':
            env.render()
            time.sleep(0.1)
            
        # 打印当前系统状态
        if step % 10 == 0:
            coverage_rate, connected, max_coverage, unconnected = env.calculate_coverage_complete()
            print(f"Step {step}: 覆盖率={coverage_rate:.2f}, 连通性={connected}")

    # 测试阶段2：UAV失效
    print("\n第二阶段：模拟UAV失效")
    failed_uav = 2
    env.fail_uav(failed_uav)
    print(f"UAV {failed_uav} 已失效")

    for step in range(50):
        # 只为活跃的UAV生成动作
        actions = {
            f"agent_{i}": env.action_space[i].sample().astype(np.float32) 
            for i in env.active_agents  # 只为活跃的UAV生成动作
        }
        
        obs, rewards, dones, _, _ = env.step(actions)
        
        if render_mode == 'human':
            env.render()
            time.sleep(0.1)
        
        if step % 10 == 0:
            coverage_rate, connected, max_coverage, unconnected = env.calculate_coverage_complete()
            print(f"Step {step}: 覆盖率={coverage_rate:.2f}, 连通性={connected}, 未连接UAV数={unconnected}")

    # 测试阶段3：添加新UAV
    print("\n第三阶段：添加新UAV")
    new_uav_pos = [-0.8, -0.8]  # 在左下角添加新UAV
    new_uav_idx = env.add_uav(position=new_uav_pos)
    print(f"在位置 {new_uav_pos} 添加新UAV {new_uav_idx}")

    for step in range(50):  # 观察系统整合新UAV
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) for i in range(env.num_agents)}
        obs, rewards, dones, _, _ = env.step(actions)
        
        if render_mode == 'human':
            env.render()
            time.sleep(0.1)
            
        if step % 10 == 0:
            coverage_rate, connected, max_coverage, unconnected = env.calculate_coverage_complete()
            print(f"Step {step}: 覆盖率={coverage_rate:.2f}, 连通性={connected}, 最大覆盖率={max_coverage:.2f}")

    env.close()

def test_random_topology_changes(render_mode='human', total_steps=200):
    print("\n=== 测试随机拓扑变化 ===")
    
    env = UAVEnv(
        num_agents=5,
        num_targets=10,
        render_mode=render_mode
    )
    obs, _ = env.reset()
    
    change_probability = 0.01  # 每步有1%的概率发生拓扑变化
    
    for step in range(total_steps):
        # 随机触发拓扑变化
        if np.random.random() < change_probability:
            if np.random.random() < 0.5:  # 50%概率失效或添加
                # 随机选择一个UAV失效
                active_uavs = env.active_agents
                if len(active_uavs) > 3:  # 保持至少3个UAV
                    failed_uav = np.random.choice(active_uavs)
                    env.fail_uav(failed_uav)
                    print(f"\nStep {step}: UAV {failed_uav} 失效")
            else:
                # 尝试添加新UAV
                new_uav_idx = env.add_uav()
                if new_uav_idx is not None:
                    print(f"\nStep {step}: 添加新UAV {new_uav_idx}")
        
        # 正常步进
        actions = {f"agent_{i}": env.action_space[i].sample().astype(np.float32) for i in range(env.num_agents)}
        obs, rewards, dones, _, _ = env.step(actions)
        
        if render_mode == 'human':
            env.render()
            time.sleep(0.1)
            
        # 每20步打印一次状态
        if step % 20 == 0:
            coverage_rate, connected, max_coverage, unconnected = env.calculate_coverage_complete()
            print(f"Step {step}: 覆盖率={coverage_rate:.2f}, 连通性={connected}, "
                  f"活跃UAV数量={len(env.active_agents)}")
    
    env.close()

if __name__ == '__main__':
    # 测试固定场景的拓扑变化
    test_topology_changes(render_mode='human')
    
    # 测试随机拓扑变化
    # test_random_topology_changes(render_mode='human')