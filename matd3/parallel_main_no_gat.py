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
import time

# 导入本地模块
from mpe_uav.uav_env.uav_env import UAVEnv
from mpe_uav.uav_env.parallel_uav_env import ParallelUAVEnv
from matd3_no_gat import MATD3, ReplayBuffer
from utils import OUNoise
from config import CONFIG

# 获取当前文件的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 获取 main.py 文件所在的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, './output_no_gat_parallel/models/test1')  # 模型保存文件夹
video_dir = os.path.join(current_dir, './output_no_gat_parallel/videos/test1')  # 视频保存文件夹
runs_dir = os.path.join(current_dir, './output_no_gat_parallel/runs/test1')  # TensorBoard 日志文件夹

# 设置常量
FRAME_SAVE_INTERVAL = 10  # 减少视频保存频率，提高训练速度
final_times = 20  # 减少最后保存的视频数量

# 自动创建 models 和 videos 文件夹
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

# 在代码最开始添加 去掉pygame的提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def main():
    # 设置性能优化参数
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    writer = SummaryWriter(log_dir=runs_dir)
    render_mode = CONFIG["render_mode"]
    num_episodes = CONFIG["num_episodes"]
    max_steps_per_episode = CONFIG["max_steps_per_episode"]
    initial_random_steps = CONFIG["initial_random_steps"]
    
    # 设置随机种子
    seed = random.randint(0, 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化并行环境
    num_envs = max(1, mp.cpu_count() - 2)  # 使用CPU核心数-2个环境
    print(f"使用 {num_envs} 个并行环境")
    
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

    # 初始化 MATD3
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    matd3 = MATD3(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=device,
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        policy_noise=CONFIG.get("noise_std", 0.2),
        noise_clip=CONFIG.get("noise_clip", 0.5),
        policy_delay=CONFIG.get("policy_delay", 2)
    )

    # 初始化或加载模型
    model_path = os.path.join(model_dir, "matd3_model_episode_4000.pth")
    if os.path.exists(model_path):
        print("加载已保存的模型...")
        try:
            matd3.load(model_path)
            print("模型加载完成。")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("初始化新模型。")
            
    # 初始化 ReplayBuffer
    replay_buffer = ReplayBuffer(
        buffer_size=CONFIG["buffer_size"], 
        batch_size=CONFIG["batch_size"], 
        device=device
    )

    # 动态噪声设置
    noise_std = CONFIG.get("noise_std", 0.2)
    noise_decay_rate = CONFIG.get("noise_decay_rate", 0.9995)

    total_rewards = []
    actor_losses = []
    critic_losses = []
    training_start_time = time.time()

    # 初始填充经验回放缓冲区
    print(f"初始随机采样 {initial_random_steps} 步...")
    obs_dict_list, _ = env.reset()
    
    # 计算每个环境需要采样的步数
    steps_per_env = initial_random_steps // num_envs
    
    for step in tqdm(range(steps_per_env)):
        # 为所有环境准备随机动作
        actions_per_env = []
        for env_idx in range(env.num_envs):
            env_actions = {}
            for agent in agents:
                if agent in obs_dict_list[env_idx]:
                    env_actions[agent] = np.random.uniform(-1, 1, env.get_action_space(agent).shape)
            actions_per_env.append(env_actions)

        # 并行步进所有环境
        next_obs_list, rewards_list, dones_list, truncated_list, infos_list = env.step(actions_per_env)

        # 将所有环境的经验添加到replay buffer
        for env_idx in range(env.num_envs):
            all_agents = set(agents) | set(obs_dict_list[env_idx].keys()) | set(next_obs_list[env_idx].keys())
            obs_filled = {agent: obs_dict_list[env_idx].get(agent, np.zeros(obs_dim)) for agent in all_agents}
            actions_filled = {agent: actions_per_env[env_idx].get(agent, np.zeros(action_dim)) for agent in all_agents}
            rewards_filled = {agent: rewards_list[env_idx].get(agent, 0.0) for agent in all_agents}
            next_obs_filled = {agent: next_obs_list[env_idx].get(agent, np.zeros(obs_dim)) for agent in all_agents}
            dones_filled = {agent: dones_list[env_idx].get(agent, True) for agent in all_agents}
            
            replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)

        obs_dict_list = next_obs_list
        
        # 如果所有环境都结束，重置环境
        all_done = all([all(dones.values()) for dones in dones_list])
        if all_done:
            obs_dict_list, _ = env.reset()
    
    print("开始正式训练...")
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        episode_seed = random.randint(0, 1000)
        record_video = (episode % FRAME_SAVE_INTERVAL == 0) or (episode >= num_episodes - final_times)
        
        obs_dict_list, _ = env.reset(seed=episode_seed)
        episode_rewards = [0] * env.num_envs
        frames = []
        max_coverage_rates = [0] * env.num_envs
        
        # 动态调整噪声
        current_noise = noise_std * (noise_decay_rate ** episode)
        
        episode_start_time = time.time()

        for step in range(max_steps_per_episode):
            # 为所有环境准备动作
            actions_per_env = []
            for env_idx in range(env.num_envs):
                # 准备当前环境的观测
                env_obs = obs_dict_list[env_idx]
                # 使用matd3的select_action方法获取动作
                env_actions = matd3.select_action(env_obs, noise=current_noise)
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
                loss_info = matd3.train(replay_buffer)
                actor_loss = loss_info['actor_loss']
                critic_loss = loss_info['critic_loss']
                
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
        writer.add_scalar('Noise Level', current_noise, episode)
        
        episode_time = time.time() - episode_start_time
        total_time = time.time() - training_start_time
        
        total_rewards.append(avg_episode_reward)
        
        # 只在特定间隔打印详细信息，减少输出
        if episode % 10 == 0 or episode < 10:
            print(f"\n回合 {episode + 1}/{num_episodes} 完成 (用时: {episode_time:.2f}秒, 总时间: {total_time/60:.2f}分钟):")
            print(f"平均奖励: {avg_episode_reward:.2f}")
            print(f"平均最大覆盖率: {avg_max_coverage_rate*100:.2f}%")
            print(f"平均覆盖率: {avg_coverage_rate * 100:.2f}%")
            print(f"噪声水平: {current_noise:.4f}")
            if actor_losses and critic_losses:
                print(f"Actor损失: {actor_losses[-1]:.4f}, Critic损失: {critic_losses[-1]:.4f}")

        # 每隔500轮保存一次模型，减少IO操作
        if (episode + 1) % 500 == 0:
            intermediate_model_path = os.path.join(model_dir, f"matd3_model_episode_{episode + 1}.pth")
            try:
                matd3.save(intermediate_model_path)
                print(f"模型已保存到 {intermediate_model_path}")
            except Exception as e:
                print(f"保存模型时出错: {e}")

        # 保存视频
        if record_video and frames:
            try:
                video_path = os.path.join(video_dir, f"{episode + 1}_{int(avg_episode_reward)}_{avg_max_coverage_rate* 100:.2f}%_{avg_coverage_rate * 100:.2f}%.mp4")
                with imageio.get_writer(video_path, fps=10) as video:
                    for frame in frames:
                        video.append_data(frame)
                print(f"视频已保存到 {video_path}")
            except Exception as e:
                print(f"保存视频时出错: {e}")

    # 保存最终模型
    model_save_path = os.path.join(model_dir, f"matd3_model_episode_{num_episodes}.pth")
    matd3.save(model_save_path)
    print(f"最终模型已保存到 {model_save_path}")
    
    # 打印总训练时间
    total_training_time = time.time() - training_start_time
    print(f"总训练时间: {total_training_time/60:.2f}分钟 ({total_training_time/3600:.2f}小时)")

    writer.close()
    env.close()

if __name__ == "__main__":
    main()