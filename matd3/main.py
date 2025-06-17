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

# 导入本地模块
from mpe_uav.uav_env.uav_env import UAVEnv
from matd3_no_gat import MATD3, PrioritizedReplayBuffer
from utils import OUNoise
from config import CONFIG

# 获取当前文件的父目录（matd3_master 所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 获取 coverage_master_v3 文件夹所在目录

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

def main():
    writer = SummaryWriter(log_dir=runs_dir)  # TensorBoard 日志文件夹
    render_mode = CONFIG["render_mode"]
    num_episodes = 10000
    max_steps_per_episode = 128
    seed = random.randint(0, 1000)  # 随机种子 全部随机

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)  # 也设置Python的随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 初始化环境并设置为 human 模式
    env = UAVEnv(render_mode=render_mode)

    obs, _ = env.reset(seed=seed)
    # 获取智能体列表
    agents = env.agents

    # 获取观测和动作空间维度
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]

    # 创建观测和动作维度的字典
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}

    # 初始化或加载 MATD3
    model_path = os.path.join(model_dir, "matd3_model_episode_4000.pth")
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

    # 确保所有模型组件在 GPU 上
    maddpg.to(device)
    for agent in agents:
        maddpg.actors[agent].to(device)
        maddpg.critics1[agent].to(device)  # 第一个 Critic
        maddpg.critics2[agent].to(device)  # 第二个 Critic


    """
    加载模型
    """
    if os.path.exists(model_path):
        print("加载已保存的模型...")
        checkpoint = torch.load(model_path, map_location=device)
        # 加载模型权重
        maddpg.load_state_dict(checkpoint['model_state_dict'])
        # 加载 ReplayBuffer 状态
        replay_buffer = checkpoint.get('replay_buffer', PrioritizedReplayBuffer(
            buffer_size=CONFIG["buffer_size"], batch_size=CONFIG["batch_size"], device=device)
        )
        # 加载优化器状态
        for agent in agents:
            if agent in maddpg.actor_optimizers:
                maddpg.actor_optimizers[agent].load_state_dict(checkpoint['actor_optimizer_state_dict'][agent])
            if agent in maddpg.critic_optimizers1:
                maddpg.critic_optimizers1[agent].load_state_dict(checkpoint['critic_optimizer1_state_dict'][agent])
            if agent in maddpg.critic_optimizers2:
                maddpg.critic_optimizers2[agent].load_state_dict(checkpoint['critic_optimizer2_state_dict'][agent])
        # 确保模型在正确的设备上
        maddpg.to(device)
        print("完整模型加载完成。")
    else:
        # 如果没有保存的模型，初始化模型到设备
        replay_buffer = PrioritizedReplayBuffer(buffer_size=CONFIG["buffer_size"], batch_size=CONFIG["batch_size"], device=device)
        print("未找到保存的模型，初始化新模型。")

    # 初始化噪声
    noise = {agent: OUNoise(action_dim) for agent in agents}

    total_rewards = []
    actor_losses = []
    critic_losses = []

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        # 随机种子
        seed = random.randint(0, 1000)  # 每一轮都设置一个新的随机种子
        # 每隔 FRAME_SAVE_INTERVAL 轮保存一次视频，还有最后 final_times 轮保存视频
        record_video = (episode % (FRAME_SAVE_INTERVAL * 5) == 0) or (episode >= num_episodes - final_times)  # 减少视频保存频率
        obs, _ = env.reset(seed=seed)
        episode_reward = 0
        frames = []
        max_coverage_rate = 0  # 每一轮重置最大覆盖率

        print(f"\nStarting episode {episode + 1}")
        step_count = 0
        
        for step in range(max_steps_per_episode):
            step_count += 1
            actions = {}
            for agent in agents:
                if agent not in obs:
                    continue
                observation = obs[agent]
                # 批量处理观测
                observations = {agent: observation}
                observation_tensors = torch.from_numpy(np.stack(list(observations.values()))).float().to(device)
                with torch.no_grad():
                    try:
                        action_tensors = {agent: maddpg.actors[agent](observation_tensors[i:i+1]) 
                                        for i, agent in enumerate(observations.keys())}
                        action = action_tensors[agent].cpu().numpy().squeeze(0)
                        action += noise[agent].noise()
                        actions[agent] = np.clip(action, -1, 1)  # 动作范围限制在 -1 到 1 之间
                    except Exception as e:
                        print(f"Error in action computation for agent {agent}: {e}")
                        continue

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
                print(f"Error adding to replay buffer: {e}")
                continue

            # 更新 MATD3 网络
            actor_loss, critic_loss = maddpg.update(replay_buffer)
            if actor_loss is not None and critic_loss is not None:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                if step_count % 100 == 0:  # 每100步打印一次损失
                    print(f"Step {step_count}: Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            # 如果所有智能体都完成，则提前结束
            if all([dones.get(agent, True) for agent in agents]):
                print(f"Episode ended early at step {step_count}")
                break

            obs = next_obs

            # 保存当前帧到视频（根据 record_video）
            if record_video and step:  # 每步都检查
                frame = env.render()
                frame = cv2.resize(frame, (704, 704))
                frames.append(frame)

        # 计算并记录覆盖率
        coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav = env.calculate_coverage_complete()
        max_coverage_rate = max(coverage_rate, max_coverage_rate)

        # 将覆盖率记录到 TensorBoard
        writer.add_scalar('Coverage Rate', coverage_rate, episode)      # 正常的覆盖率
        writer.add_scalar('Max Coverage Rate', max_coverage_rate, episode)  # 最大的覆盖率
        writer.add_scalar('Unconnected Uav', unconnected_uav, episode)  # 最大的覆盖率

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} completed:")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Max Coverage Rate: {max_coverage_rate*100:.2f}%")
        print(f"Coverage Rate: {coverage_rate * 100:.2f}%")
        print(f"Steps taken: {step_count}")

        # TensorBoard 可视化
        writer.add_scalar('Total Reward', episode_reward, episode)
        if actor_loss is not None and critic_loss is not None:
            writer.add_scalar('Actor Loss', actor_loss, episode)
            writer.add_scalar('Critic Loss', critic_loss, episode)

        # 每隔1000轮保存一次完整模型
        if (episode + 1) % 1000 == 0:
            intermediate_model_path = os.path.join(model_dir, f"matd3_model_episode_{episode + 1}.pth")
            try:
                torch.save({
                    'model_state_dict': maddpg.state_dict(),
                    'replay_buffer': replay_buffer,  # 保存 ReplayBuffer 状态
                    'actor_optimizer_state_dict': {agent: maddpg.actor_optimizers[agent].state_dict() for agent in agents},
                    'critic_optimizer1_state_dict': {agent: maddpg.critic_optimizers1[agent].state_dict() for agent in agents},
                    'critic_optimizer2_state_dict': {agent: maddpg.critic_optimizers2[agent].state_dict() for agent in agents},
                    'config': CONFIG,  # 保存配置文件
                }, intermediate_model_path)
                print(f"完整模型已保存到 {intermediate_model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

        # 每隔视频保存间隔保存一次视频
        if record_video and frames:
            try:
                video_path = os.path.join(video_dir, f"{episode + 1}_{int(total_rewards[episode])}_{max_coverage_rate* 100:.2f}%_{coverage_rate * 100:.2f}%.mp4")
                with imageio.get_writer(video_path, fps=10) as video:
                    for frame in frames:
                        video.append_data(frame)
                print(f"视频已保存到 {video_path}")
            except Exception as e:
                print(f"Error saving video: {e}")

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