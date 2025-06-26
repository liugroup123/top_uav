# -*- coding: utf-8 -*-
"""
使用简化环境的MATD3训练脚本
基于main_no_gat.py，适配uav_env_clean.py
"""

import sys
import os

# 获取当前文件的父目录（mpe_uav目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # mpe_uav目录
sys.path.append(parent_dir)

import cv2
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

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
model_dir = os.path.join(current_dir, './output_clean_env/models/test1')  # 模型保存文件夹
video_dir = os.path.join(current_dir, './output_clean_env/videos/test1')  # 视频保存文件夹
runs_dir = os.path.join(current_dir, './output_clean_env/runs/test1')  # TensorBoard 日志文件夹

# 创建目录
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

def setup_cuda():
    """设置CUDA优化"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  使用CPU训练")

def setup_seeds(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    """主训练函数"""
    print("🚀 开始使用简化环境训练MATD3")
    
    # 设置优化和种子
    setup_cuda()
    setup_seeds()
    
    # 训练参数
    num_episodes = CONFIG["num_episodes"]
    max_steps = CONFIG["max_steps"]
    initial_random_steps = CONFIG["initial_random_steps"]
    render_mode = None  # 'human' 或 None

    # 速度优化参数
    train_frequency = CONFIG.get("train_frequency", 3)  # 每3步训练一次
    log_interval = CONFIG.get("log_interval", 10)       # 每10步记录一次日志
    
    # 创建简化环境
    env = UAVEnv(
        render_mode=render_mode,
        experiment_type='probabilistic',  # 使用概率驱动拓扑变化
        num_agents=5,
        num_targets=10,
        max_steps=max_steps,
        min_active_agents=4,
        max_active_agents=6
    )
    
    print(f"✅ 环境创建成功")
    print(f"环境类型: {env.__class__.__name__}")
    print(f"实验模式: {env.experiment_type}")
    print(f"UAV数量: {env.num_agents}")
    print(f"目标数量: {env.num_targets}")
    
    # 获取环境信息
    obs, _ = env.reset()
    agents = env.agents
    
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]
    
    print(f"观察维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"智能体数量: {len(agents)}")
    
    # 创建MATD3算法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    
    matd3 = MATD3(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=device,
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"]
    )
    
    # 创建经验回放
    replay_buffer = ReplayBuffer(
        buffer_size=CONFIG["buffer_size"],
        batch_size=CONFIG["batch_size"],
        device=device
    )
    
    # 创建TensorBoard记录器
    writer = SummaryWriter(runs_dir)

    print(f"📊 TensorBoard日志: {runs_dir}")
    print(f"💾 模型保存目录: {model_dir}")
    print(f"📹 视频保存目录: {video_dir}")
    
    # 初始随机采样
    print(f"🎲 开始初始随机采样 ({initial_random_steps} 步)...")
    obs, _ = env.reset()
    
    for step in tqdm(range(initial_random_steps)):
        # 完全随机动作
        actions = {agent: np.random.uniform(-1, 1, env.get_action_space(agent).shape) 
                  for agent in agents if agent in obs}
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        
        # 将经验添加到 Replay Buffer
        all_agents = set(agents) | set(obs.keys()) | set(next_obs.keys())
        obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
        rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
        next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
        dones_filled = {agent: dones.get(agent, True) for agent in all_agents}
        
        replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
        obs = next_obs
        
        # 如果episode结束，重置环境
        if any(dones.values()):
            obs, _ = env.reset()
    
    print(f"✅ 初始采样完成，缓冲区大小: {len(replay_buffer)}")
    
    # 开始训练
    print(f"🎯 开始训练 ({num_episodes} episodes)...")
    
    # 训练统计
    episode_rewards = []
    episode_coverages = []
    max_coverage_rate = 0.0
    
    # 噪声参数
    noise_std = CONFIG["noise_std"]
    noise_decay = CONFIG["noise_decay"]
    min_noise = CONFIG["min_noise"]
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_coverage_history = []
        
        # 当前噪声
        current_noise = max(min_noise, noise_std * (noise_decay ** episode))
        
        # 视频录制（每100个episode录制一次）
        frames = []
        record_video = (episode % 100 == 0) and render_mode is None
        
        for step in range(max_steps):
            # 使用 MATD3 的 select_action 方法选择动作
            actions = matd3.select_action(obs, noise=current_noise)

            next_obs, rewards, dones, truncated, infos = env.step(actions)
            episode_reward += sum(rewards.values())

            # 将经验添加到 Replay Buffer
            all_agents = set(agents) | set(obs.keys()) | set(next_obs.keys())
            obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
            rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
            next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            dones_filled = {agent: dones.get(agent, True) for agent in all_agents}

            replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)

            # 训练网络（优化训练频率）
            if len(replay_buffer) > CONFIG["batch_size"] and step % train_frequency == 0:
                # 批量训练提高效率
                loss_info = None
                for _ in range(2):  # 每次触发训练2轮
                    loss_info = matd3.train(replay_buffer)

                # 记录损失到TensorBoard（减少写入频率）
                if loss_info and step % log_interval == 0:
                    writer.add_scalar('Loss/Actor', loss_info['actor_loss'], episode * max_steps + step)
                    writer.add_scalar('Loss/Critic', loss_info['critic_loss'], episode * max_steps + step)

            obs = next_obs

            # 录制视频帧
            if record_video:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)

        # 计算并记录覆盖率
        coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav = env.calculate_coverage_complete()
        max_coverage_rate = max(coverage_rate, max_coverage_rate)

        # 将覆盖率记录到 TensorBoard
        writer.add_scalar('Performance/Coverage_Rate', coverage_rate, episode)
        writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Performance/Max_Coverage_Rate', max_coverage_rate, episode)
        writer.add_scalar('Training/Noise_Std', current_noise, episode)
        writer.add_scalar('Training/Active_UAVs', len(env.active_agents), episode)

        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_coverages.append(coverage_rate)

        # 保存视频
        if record_video and frames:
            video_path = f"{video_dir}/episode_{episode}.mp4"
            save_video(frames, video_path)

        # 定期保存模型
        if episode % 100 == 0:
            model_save_path = f"{model_dir}/matd3_episode_{episode}.pth"
            matd3.save(model_save_path)
            print(f"💾 模型已保存: {model_save_path}")

        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_coverage = np.mean(episode_coverages[-50:])
            print(f"Episode {episode}: 平均奖励={avg_reward:.2f}, 平均覆盖率={avg_coverage:.3f}, 噪声={current_noise:.3f}")

    # 保存最终模型
    final_model_path = f"{model_dir}/matd3_final.pth"
    matd3.save(final_model_path)
    print(f"🎉 训练完成！最终模型保存至: {final_model_path}")

    # 关闭环境和记录器
    env.close()
    writer.close()

def save_video(frames, path, fps=30):
    """保存视频"""
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

if __name__ == '__main__':
    main()
