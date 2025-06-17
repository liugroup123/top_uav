# -*- coding: utf-8 -*-
import sys
import os

# 获取项目根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import cv2
import random
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import imageio

# 现在导入本地模块
from mpe_uav.uav_env.uav_env import UAVEnv
from mpe_uav.uav_env.parallel_uav_env import ParallelUAVEnv
from matd3 import MATD3, PrioritizedReplayBuffer
from utils import OUNoise
from config import CONFIG

# 获取当前文件的父目录（matd3_master 所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 获取 coverage_master_v3 文件夹所在目录

# 将父目录添加到模块搜索路径
sys.path.append(parent_dir)

# 获取 main.py 文件所在的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, './output/models/test1')  # 模型保存文件夹
video_dir = os.path.join(current_dir, './output/videos/test1')  # 视频保存文件夹
runs_dir = os.path.join(current_dir, './output/runs/test1')  # TensorBoard 日志文件夹
#日志文件夹

FRAME_SAVE_INTERVAL = 1  # 每100轮保存视频
final_times = 50

# 自动创建 models 和 videos 文件夹
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

# 在代码最开始添加 去掉pygame的提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def main():
    writer = SummaryWriter(log_dir=runs_dir)
    render_mode = CONFIG["render_mode"]
    num_episodes = 10000
    max_steps_per_episode = 128
    seed = random.randint(0, 1000)

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化并行环境
    num_envs = 1  # 使用CPU核心数-1个环境
    env = ParallelUAVEnv(
        num_envs=num_envs,
        num_agents=5,
        num_targets=10,
        render_mode="rgb_array"
    )

    obs_list, _ = env.reset(seed=seed)
    agents = env.agents

    # 获取观测和动作空间维度
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]

    # 初始化 MATD3 (保持不变)
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    maddpg = MATD3(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        batch_size=CONFIG["batch_size"],
        policy_delay=CONFIG.get("policy_delay", 2),
        noise_std=CONFIG.get("noise_std", 0.2),
        noise_clip=CONFIG.get("noise_clip", 0.5),
    )

    # 初始化replay buffer
    replay_buffer = PrioritizedReplayBuffer(
        buffer_size=CONFIG["buffer_size"], 
        batch_size=CONFIG["batch_size"], 
        device=device
    )

    # 确保所有模型组件在 GPU 上
    maddpg.to(device)
    for agent in agents:
        maddpg.actors[agent].to(device)
        maddpg.critics1[agent].to(device)  # 第一个 Critic
        maddpg.critics2[agent].to(device)  # 第二个 Critic

    # 初始化噪声
    noise = {agent: OUNoise(action_dim) for agent in agents}

    total_rewards = []
    actor_losses = []
    critic_losses = []

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        seed = random.randint(0, 1000)
        record_video = (episode % FRAME_SAVE_INTERVAL == 0) or (episode >= num_episodes - final_times)
        
        obs_dict_list, _ = env.reset(seed=seed)
        episode_rewards = [0] * env.num_envs
        frames = []
        max_coverage_rates = [0] * env.num_envs

        for step in range(max_steps_per_episode):
            # 为所有环境准备动作
            actions_per_env = []
            for env_idx in range(env.num_envs):
                env_actions = {}
                for agent in agents:
                    if agent not in obs_dict_list[env_idx]:
                        continue
                    observation = obs_dict_list[env_idx][agent]
                    observation_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_tensor = maddpg.actors[agent](observation_tensor)
                    action = action_tensor.cpu().numpy().squeeze(0)
                    action += noise[agent].noise()
                    env_actions[agent] = np.clip(action, -1, 1)
                actions_per_env.append(env_actions)

            # 并行步进所有环境
            next_obs_list, rewards_list, dones_list, truncated_list, infos_list = env.step(actions_per_env)

            # 更新累积奖励
            for env_idx in range(env.num_envs):
                episode_rewards[env_idx] += sum(rewards_list[env_idx].values())

            # 将所有环境的经验添加到replay buffer
            for env_idx in range(env.num_envs):
                all_agents = set(agents) | set(obs_dict_list[env_idx].keys()) | set(next_obs_list[env_idx].keys())
                obs_filled = {agent: obs_dict_list[env_idx].get(agent, np.zeros(obs_dim)) for agent in all_agents}
                actions_filled = {agent: actions_per_env[env_idx].get(agent, np.zeros(action_dim)) for agent in all_agents}
                rewards_filled = {agent: rewards_list[env_idx].get(agent, 0.0) for agent in all_agents}
                next_obs_filled = {agent: next_obs_list[env_idx].get(agent, np.zeros(obs_dim)) for agent in all_agents}
                dones_filled = {agent: dones_list[env_idx].get(agent, True) for agent in all_agents}
                
                replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)

            # 更新MATD3网络
            if len(replay_buffer) > CONFIG["batch_size"]:
                actor_loss, critic_loss = maddpg.update(replay_buffer)
                if actor_loss is not None and critic_loss is not None:
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

            obs_dict_list = next_obs_list

            # 只记录第一个环境的视频（如果需要）
            if record_video and step % 2 == 0:
                frame = env.render()  # 直接调用 render 方法
                if frame is not None:
                    frame = cv2.resize(frame, (704, 704))
                    frame = np.ascontiguousarray(frame)
                    frames.append(frame)

        # 计算所有环境的平均覆盖率和最大覆盖率
        coverage_rates = []
        max_coverage_rates = []
        unconnected_uavs = []
        for env_idx in range(env.num_envs):
            if env_idx == 0:  # 只计算第一个环境的覆盖率
                coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav = env.calculate_coverage_complete()
                coverage_rates.append(coverage_rate)
                max_coverage_rates.append(max_coverage_rate)
                unconnected_uavs.append(unconnected_uav)

        # 记录平均指标
        avg_coverage_rate = np.mean(coverage_rates)
        avg_max_coverage_rate = np.mean(max_coverage_rates)
        avg_unconnected_uav = np.mean(unconnected_uavs)
        avg_episode_reward = np.mean(episode_rewards)

        # TensorBoard记录
        writer.add_scalar('Coverage Rate', avg_coverage_rate, episode)
        writer.add_scalar('Max Coverage Rate', avg_max_coverage_rate, episode)
        writer.add_scalar('Unconnected Uav', avg_unconnected_uav, episode)
        writer.add_scalar('Total Reward', avg_episode_reward, episode)

        print(f"Episode:{episode + 1}/{num_episodes}, "
              f"Avg Reward: {avg_episode_reward:.2f}, "
              f"Avg Max Rate:{avg_max_coverage_rate*100:.2f}%, "
              f"Avg Coverage Rate: {avg_coverage_rate * 100:.2f}%")

        # 保存模型和视频的代码保持不变...

    # 保存最终模型
    model_save_path = os.path.join(model_dir, f"matd3_model_episode_{num_episodes}.pth")
    torch.save({
        'model_state_dict': maddpg.state_dict(),
        'replay_buffer': replay_buffer,
        'actor_optimizer_state_dict': {agent: maddpg.actor_optimizers[agent].state_dict() for agent in agents},
        'critic_optimizer1_state_dict': {agent: maddpg.critic_optimizers1[agent].state_dict() for agent in agents},
        'critic_optimizer2_state_dict': {agent: maddpg.critic_optimizers2[agent].state_dict() for agent in agents},
        'config': CONFIG,
    }, model_save_path)
    print(f"最终完整模型已保存到 {model_save_path}")

    writer.close()
    env.close()

if __name__ == "__main__":
    main()