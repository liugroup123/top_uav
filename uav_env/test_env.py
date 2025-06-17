import time
import numpy as np
import os,sys
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from mpe_uav.uav_env.uav_env import UAVEnv
import pdb
def test_env(render_mode):
    print(f"\nTesting UAVEnv with render_mode='{render_mode}'")
    env = UAVEnv(render_mode=render_mode)
    obs,_= env.reset()

    # # 打印观测维度（使用字典结构）
    # print("Initial observation shapes:", {k: v.shape for k, v in obs.items()})

    # # 打印动作空间维度（每个 agent）
    # print("Action space shapes:", {f"agent_{i}": env.action_space[i].shape for i in range(env.num_agents)})

    for step in range(100):
        # 构建动作字典
        actions = {f"agent_{i}": env.action_space[i].sample() for i in range(env.num_agents)}

        obs, rewards, dones, terminated, infos = env.step(actions)
        # print(f"Step {step}: rewards={rewards}, dones={dones}")

        # 渲染
        if render_mode == 'human':
            env.render()
            time.sleep(0.1)
        elif render_mode == 'rgb_array':
            frame = env.render()
            print(f"RGB frame shape: {frame.shape}")

        if all(dones.values()):
            break

    env.close()

if __name__ == '__main__':
    test_env(render_mode='human')
    # test_env(render_mode='rgb_array')
