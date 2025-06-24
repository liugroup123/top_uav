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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import imageio
import time

# 导入本地模块
from uav_top_env.uav_env_top import UAVEnv
from matd3_no_gat import MATD3, ReplayBuffer
from utils import OUNoise
from config import CONFIG

# 获取当前文件的父目录（matd3_master 所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 获取 coverage_master_v3 文件夹所在目录

# 获取 main.py 文件所在的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, './output_no_gat/models/test1')  # 模型保存文件夹
video_dir = os.path.join(current_dir, './output_no_gat/videos/test1')  # 视频保存文件夹
runs_dir = os.path.join(current_dir, './output_no_gat/runs/test1')  # TensorBoard 日志文件夹
#日志文件夹

FRAME_SAVE_INTERVAL = 10  # 减少视频保存频率，提高训练速度
final_times = 20  # 减少最后保存的视频数量

# 自动创建 models 和 videos 文件夹
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)

def main():
    # 设置性能优化参数
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    writer = SummaryWriter(log_dir=runs_dir)  # TensorBoard 日志文件夹
    render_mode = CONFIG["render_mode"]
    num_episodes = CONFIG["num_episodes"]
    max_steps_per_episode = CONFIG["max_steps_per_episode"]
    initial_random_steps = CONFIG["initial_random_steps"]  # 初始随机步骤
    
    # 设置随机种子
    seed = random.randint(0, 1000)  # 随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)  # 也设置Python的随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化环境
    env = UAVEnv(
        render_mode=render_mode,
        experiment_type='normal',  # 可以改为 'uav_loss', 'uav_addition', 'random_mixed'
        num_agents=5,
        num_targets=10
    )

    obs, _ = env.reset(seed=seed)
    # 获取智能体列表
    agents = env.agents

    # 获取观测和动作空间维度
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]

    # 创建观测和动作维度的字典
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}

    # 初始化 MATD3
    model_path = os.path.join(model_dir, "matd3_model_episode_4000.pth")
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

    # 初始化噪声
    noise_std = CONFIG.get("noise_std", 0.1)
    noise_decay_rate = CONFIG.get("noise_decay_rate", 0.995)

    total_rewards = []
    actor_losses = []
    critic_losses = []
    training_start_time = time.time()

    # 初始填充经验回放缓冲区
    print(f"初始随机采样 {initial_random_steps} 步...")
    obs, _ = env.reset()
    for step in tqdm(range(initial_random_steps)):
        # 完全随机动作
        actions = {agent: np.random.uniform(-1, 1, env.get_action_space(agent).shape) for agent in agents if agent in obs}
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
        if all([dones.get(agent, True) for agent in agents]):
            obs, _ = env.reset()
    
    print("开始正式训练...")
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        # 随机种子
        episode_seed = random.randint(0, 1000)  # 每一轮都设置一个新的随机种子
        # 每隔 FRAME_SAVE_INTERVAL 轮保存一次视频，还有最后 final_times 轮保存视频
        record_video = (episode % FRAME_SAVE_INTERVAL == 0) or (episode >= num_episodes - final_times)
        obs, _ = env.reset(seed=episode_seed)
        episode_reward = 0
        frames = []
        max_coverage_rate = 0  # 每一轮重置最大覆盖率
        
        # 动态调整噪声
        current_noise = noise_std * (noise_decay_rate ** episode)

        step_count = 0
        episode_start_time = time.time()
        
        for step in range(max_steps_per_episode):
            step_count += 1
            
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

            try:
                replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)
            except Exception as e:
                print(f"添加到经验回放缓冲区时出错: {e}")
                continue

            # 当经验回放缓冲区有足够的样本时开始训练
            if len(replay_buffer) > CONFIG["batch_size"]:
                # 更新 MATD3 网络
                loss_info = matd3.train(replay_buffer)
                actor_loss = loss_info['actor_loss']
                critic_loss = loss_info['critic_loss']
                
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # 如果所有智能体都完成，则提前结束
            if all([dones.get(agent, True) for agent in agents]):
                break

            obs = next_obs

            # 保存当前帧到视频（根据 record_video）
            if record_video and step % 2 == 0:  # 每隔2步保存一帧，减少内存使用
                frame = env.render()
                frame = cv2.resize(frame, (704, 704))
                frames.append(frame)

        # 计算并记录覆盖率
        coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav = env.calculate_coverage_complete()
        max_coverage_rate = max(coverage_rate, max_coverage_rate)

        # 将覆盖率记录到 TensorBoard
        writer.add_scalar('Coverage Rate', coverage_rate, episode)      # 正常的覆盖率
        writer.add_scalar('Max Coverage Rate', max_coverage_rate, episode)  # 最大的覆盖率
        writer.add_scalar('Unconnected Uav', unconnected_uav, episode)  # 未连接的无人机数量
        writer.add_scalar('Noise Level', current_noise, episode)        # 记录噪声水平

        total_rewards.append(episode_reward)
        episode_time = time.time() - episode_start_time
        total_time = time.time() - training_start_time
        
        # 只在特定间隔打印详细信息，减少输出
        if episode % 10 == 0 or episode < 10:
            print(f"\n回合 {episode + 1}/{num_episodes} 完成 (用时: {episode_time:.2f}秒, 总时间: {total_time/60:.2f}分钟):")
            print(f"总奖励: {episode_reward:.2f}")
            print(f"最大覆盖率: {max_coverage_rate*100:.2f}%")
            print(f"当前覆盖率: {coverage_rate * 100:.2f}%")
            print(f"步骤数: {step_count}, 噪声水平: {current_noise:.4f}")
            if actor_losses and critic_losses:
                print(f"Actor损失: {actor_losses[-1]:.4f}, Critic损失: {critic_losses[-1]:.4f}")

        # TensorBoard 可视化
        writer.add_scalar('Total Reward', episode_reward, episode)
        writer.add_scalar('Steps Per Episode', step_count, episode)
        writer.add_scalar('Episode Time', episode_time, episode)
        if actor_losses and critic_losses:
            writer.add_scalar('Actor Loss', actor_losses[-1], episode)
            writer.add_scalar('Critic Loss', critic_losses[-1], episode)

        # 每隔500轮保存一次模型，减少IO操作
        if (episode + 1) % 500 == 0:
            intermediate_model_path = os.path.join(model_dir, f"matd3_model_episode_{episode + 1}.pth")
            try:
                matd3.save(intermediate_model_path)
                print(f"模型已保存到 {intermediate_model_path}")
            except Exception as e:
                print(f"保存模型时出错: {e}")

        # 每隔视频保存间隔保存一次视频
        if record_video and frames:
            try:
                video_path = os.path.join(video_dir, f"{episode + 1}_{int(total_rewards[episode])}_{max_coverage_rate* 100:.2f}%_{coverage_rate * 100:.2f}%.mp4")
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