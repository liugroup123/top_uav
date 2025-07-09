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
import imageio  # 简化视频保存

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
model_dir = os.path.join(current_dir, './output_clean_env/models/test2')  # 模型保存文件夹
video_dir = os.path.join(current_dir, './output_clean_env/videos/test2')  # 视频保存文件夹
runs_dir = os.path.join(current_dir, './output_clean_env/runs/test2')  # TensorBoard 日志文件

# 确保相关目录存在
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
    log_interval = CONFIG.get("log_interval", 5)       # 每5步记录一次日志
    
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
    global_max_coverage_rate = 0.0  # 全局最大覆盖率（整个训练过程）
    
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
        
        # 视频录制（每50个episode录制一次，提高录制频率）
        frames = []
        record_video = (episode % 50 == 0) and render_mode is None
        
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
                try:
                    # 临时设置render_mode为rgb_array
                    original_mode = env.render_mode
                    env.render_mode = 'rgb_array'
                    frame = env.render()
                    env.render_mode = original_mode

                    if frame is not None and len(frame.shape) == 3:
                        # 确保帧格式正确
                        if frame.shape[2] == 3:  # RGB格式
                            frames.append(frame)
                        else:
                            print(f"⚠️  帧格式错误: {frame.shape}")
                    elif frame is None:
                        print(f"⚠️  渲染返回None (step {step})")
                except Exception as e:
                    print(f"❌ 渲染出错 (step {step}): {e}")

        # 计算并记录覆盖率
        final_coverage_rate, is_fully_connected, episode_max_coverage, unconnected_uav = env.calculate_coverage_complete()
        global_max_coverage_rate = max(final_coverage_rate, global_max_coverage_rate)  # 更新全局最大值

        # 获取episode信息
        episode_type = env.episode_plan['type']
        trigger_step = env.episode_plan['trigger_step'] 
        executed = env.episode_plan['executed']

        # 打印详细的episode信息
        print(f"Episode {episode:4d}: 类型={episode_type:8s} | "
              f"奖励={episode_reward:7.2f} | "
              f"最终覆盖率={final_coverage_rate:.3f} | "
              f"最大覆盖率={episode_max_coverage:.3f} | "
              f"活跃UAV={len(env.active_agents)}/{env.num_agents} | "
              f"噪声={current_noise:.3f}" +
              (f" | 触发步数={trigger_step}" if trigger_step else "") +
              (f" | 已执行" if executed else ""))

        # 将覆盖率记录到 TensorBoard
        writer.add_scalar('Performance/Coverage_Rate', final_coverage_rate, episode)
        writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Performance/Episode_Max_Coverage', episode_max_coverage, episode)  # 每轮最大覆盖率
        writer.add_scalar('Performance/Global_Max_Coverage', global_max_coverage_rate, episode)  # 全局最大覆盖率
        writer.add_scalar('Performance/Unconnected_UAVs', unconnected_uav, episode)  # 未连通UAV数量
        writer.add_scalar('Training/Noise_Std', current_noise, episode)
        writer.add_scalar('Training/Active_UAVs', len(env.active_agents), episode)

        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_coverages.append(final_coverage_rate)

        # 保存视频 - 简化方式 (参考动态环境)
        if record_video and frames:
            video_path = f"{video_dir}/{episode}_{episode_max_coverage:.2f}_{final_coverage_rate:.2f}.mp4"
            with imageio.get_writer(video_path, fps=60) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"视频已保存到 {video_path}")

        # 定期保存模型
        if episode % 500 == 0:
            model_save_path = f"{model_dir}/matd3_episode_{episode}.pth"
            gat_save_path = f"{model_dir}/gat_episode_{episode}.pth"

            matd3.save(model_save_path)
            env.save_gat_model(gat_save_path)  # 保存GAT模型

            print(f"💾 MATD3模型已保存: {model_save_path}")
            print(f"🧠 GAT模型已保存: {gat_save_path}")

        # 打印统计摘要（每100个episode）
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_coverage = np.mean(episode_coverages[-100:]) if len(episode_coverages) >= 100 else np.mean(episode_coverages)
            print(f"\n📊 Episode {episode} 统计摘要:")
            print(f"   最近100个episodes平均奖励: {avg_reward:.2f}")
            print(f"   最近100个episodes平均覆盖率: {avg_coverage:.3f}")
            print(f"   全局最大覆盖率: {global_max_coverage_rate:.3f}")
            print(f"   当前噪声水平: {current_noise:.3f}")
            print("-" * 80)

    # 保存最终模型
    final_model_path = f"{model_dir}/matd3_final.pth"
    final_gat_path = f"{model_dir}/gat_final.pth"

    matd3.save(final_model_path)
    env.save_gat_model(final_gat_path)

    print(f"🎉 训练完成！")
    print(f"💾 MATD3模型保存至: {final_model_path}")
    print(f"🧠 GAT模型保存至: {final_gat_path}")

    # 关闭环境和记录器
    env.close()
    writer.close()





if __name__ == '__main__':
    main()
