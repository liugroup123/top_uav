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
        max_coverage_rate = max(final_coverage_rate, max_coverage_rate)

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
        writer.add_scalar('Performance/Max_Coverage_Rate', max_coverage_rate, episode)
        writer.add_scalar('Training/Noise_Std', current_noise, episode)
        writer.add_scalar('Training/Active_UAVs', len(env.active_agents), episode)

        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_coverages.append(final_coverage_rate)

        # 保存视频
        if record_video and frames:
            video_path = f"{video_dir}/episode_{episode}.mp4"
            save_video(frames, video_path, fps=60)  # 提高帧率到60fps

        # 定期保存模型
        if episode % 500 == 0:
            model_save_path = f"{model_dir}/matd3_episode_{episode}.pth"
            matd3.save(model_save_path)
            print(f"💾 模型已保存: {model_save_path}")

        # 打印统计摘要（每100个episode）
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_coverage = np.mean(episode_coverages[-100:]) if len(episode_coverages) >= 100 else np.mean(episode_coverages)
            print(f"\n📊 Episode {episode} 统计摘要:")
            print(f"   最近100个episodes平均奖励: {avg_reward:.2f}")
            print(f"   最近100个episodes平均覆盖率: {avg_coverage:.3f}")
            print(f"   当前最大覆盖率: {max_coverage_rate:.3f}")
            print(f"   当前噪声水平: {current_noise:.3f}")
            print("-" * 80)

    # 保存最终模型
    final_model_path = f"{model_dir}/matd3_final.pth"
    matd3.save(final_model_path)
    print(f"🎉 训练完成！最终模型保存至: {final_model_path}")

    # 关闭环境和记录器
    env.close()
    writer.close()

def save_video(frames, path, fps=60):
    """保存视频 - 高质量版本"""
    if not frames:
        print(f"⚠️  没有帧数据，跳过视频保存: {path}")
        return

    try:
        # 检查帧格式
        first_frame = frames[0]
        if first_frame is None:
            print(f"❌ 第一帧为None，无法保存视频: {path}")
            return

        height, width, channels = first_frame.shape
        print(f"📹 保存视频: {path} ({width}x{height}, {len(frames)}帧, {fps}fps)")

        # 优先使用高质量编码器
        codecs_to_try = [
            ('mp4v', '.mp4'),      # 最兼容
            ('XVID', '.avi'),      # 高质量
            ('MJPG', '.avi'),      # 无损
            ('H264', '.mp4'),      # 现代编码器
            ('X264', '.mp4')       # 备选
        ]

        success = False
        for codec, ext in codecs_to_try:
            try:
                # 根据编码器调整文件扩展名
                if not path.endswith(ext):
                    adjusted_path = path.rsplit('.', 1)[0] + ext
                else:
                    adjusted_path = path

                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(adjusted_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    continue

                # 写入所有帧
                for i, frame in enumerate(frames):
                    if frame is not None and frame.shape == (height, width, channels):
                        # 确保帧是uint8格式
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                        # 转换颜色格式 RGB -> BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    else:
                        print(f"⚠️  跳过无效帧 {i}")

                out.release()

                # 检查文件是否成功创建
                if os.path.exists(adjusted_path) and os.path.getsize(adjusted_path) > 0:
                    print(f"✅ 视频保存成功: {adjusted_path} (编码器: {codec})")
                    success = True
                    break
                else:
                    print(f"❌ 编码器 {codec} 失败")

            except Exception as e:
                print(f"❌ 编码器 {codec} 出错: {e}")
                continue

        if not success:
            print(f"❌ 所有编码器都失败，无法保存视频: {path}")

    except Exception as e:
        print(f"❌ 视频保存出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
