# -*- coding: utf-8 -*-

"""
测试代码：

# 基本测试（使用最新模型，测试10个回合）
python mpe_uav/matd3/test_parallel_no_gat.py

# 指定模型路径
python mpe_uav/matd3/test_parallel_no_gat.py --model_path path/to/your/model.pth

# 测试更多回合
python mpe_uav/matd3/test_parallel_no_gat.py --episodes 20

# 指定并行环境数量
python mpe_uav/matd3/test_parallel_no_gat.py --num_envs 4

# 渲染环境（实时显示）
python mpe_uav/matd3/test_parallel_no_gat.py --render

# 保存视频
python mpe_uav/matd3/test_parallel_no_gat.py --save_video

# 指定随机种子
python mpe_uav/matd3/test_parallel_no_gat.py --seed 42
"""

import sys
import os

# 获取项目根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import cv2
import numpy as np
import torch
import imageio
import time
import argparse
from tqdm import tqdm
import multiprocessing as mp

# 导入本地模块
from mpe_uav.uav_env.uav_env import UAVEnv
from mpe_uav.uav_env.parallel_uav_env import ParallelUAVEnv
from matd3_no_gat import MATD3
from config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="测试MATD3模型（并行环境版本）")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径，如果不指定，将使用默认路径")
    parser.add_argument("--episodes", type=int, default=10, help="测试回合数")
    parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量，默认为CPU核心数-2")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--save_video", action="store_true", help="是否保存视频")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else int(time.time()) % 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置渲染模式
    render_mode = "rgb_array" if args.render or args.save_video else "None"
    
    # 设置并行环境数量
    num_envs = args.num_envs if args.num_envs is not None else max(1, mp.cpu_count() - 2)
    print(f"使用 {num_envs} 个并行环境进行评估")
    
    # 初始化并行环境
    env = ParallelUAVEnv(
        num_envs=num_envs,
        num_agents=5,
        num_targets=10,
        render_mode=render_mode
    )
    
    # 获取智能体列表和观测动作空间
    obs_list, _ = env.reset(seed=seed)
    agents = env.agents
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]
    
    # 创建观测和动作维度的字典
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    
    # 初始化MATD3模型
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
    
    # 加载模型
    if args.model_path:
        model_path = args.model_path
    else:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, './output_no_gat_parallel/models/test1')
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.startswith("matd3_model_episode_")]
        if not model_files:
            raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")
        # 按照回合数排序
        model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        model_path = os.path.join(model_dir, model_files[-1])
    
    print(f"加载模型: {model_path}")
    matd3.load(model_path)
    
    # 准备视频保存
    if args.save_video:
        video_dir = os.path.join(os.path.dirname(model_path), "../videos/test")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"test_{int(time.time())}.mp4")
        print(f"视频将保存到: {video_path}")
    
    # 测试统计
    all_rewards = []
    all_coverage_rates = []
    all_max_coverage_rates = []
    all_unconnected_uavs = []
    
    # 开始测试
    print(f"开始测试 {args.episodes} 个回合...")
    for episode in tqdm(range(args.episodes)):
        # 每个环境使用不同的种子
        episode_seeds = [seed + episode * num_envs + i for i in range(num_envs)]
        obs_dict_list, _ = env.reset(seed=episode_seeds[0])  # 只需要设置第一个环境的种子
        
        episode_rewards = [0] * num_envs
        frames = []
        
        while True:
            # 为所有环境准备动作（无噪声）
            actions_per_env = []
            for env_idx in range(num_envs):
                env_obs = obs_dict_list[env_idx]
                env_actions = matd3.select_action(env_obs, noise=0.0)
                actions_per_env.append(env_actions)
            
            # 执行动作
            next_obs_list, rewards_list, dones_list, truncated_list, infos_list = env.step(actions_per_env)
            
            # 更新累积奖励
            for env_idx in range(num_envs):
                episode_rewards[env_idx] += sum(rewards_list[env_idx].values())
            
            # 保存帧（只保存第一个环境）
            if args.save_video:
                frame = env.render()  # 渲染第一个环境
                if frame is not None:
                    frame = cv2.resize(frame, (704, 704))
                    frame = np.ascontiguousarray(frame)
                    frames.append(frame)
            elif args.render:
                env.render()
            
            # 检查是否所有环境都结束
            all_done = all([all(dones.values()) for dones in dones_list])
            if all_done:
                break
                
            obs_dict_list = next_obs_list
        
        # 计算每个环境的覆盖率
        coverage_rates = []
        max_coverage_rates = []
        unconnected_uavs = []
        
        # 只计算第一个环境的覆盖率（用于可视化）
        coverage_rate, is_fully_connected, max_coverage_rate, unconnected_uav = env.calculate_coverage_complete()
        coverage_rates.append(coverage_rate)
        max_coverage_rates.append(max_coverage_rate)
        unconnected_uavs.append(unconnected_uav)
        
        # 记录统计信息
        all_rewards.extend(episode_rewards)
        all_coverage_rates.extend(coverage_rates)
        all_max_coverage_rates.extend(max_coverage_rates)
        all_unconnected_uavs.extend(unconnected_uavs)
        
        # 打印当前回合信息
        print(f"回合 {episode+1} 结果:")
        print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  覆盖率: {coverage_rates[0]*100:.2f}%")
        print(f"  最大覆盖率: {max_coverage_rates[0]*100:.2f}%")
        print(f"  未连接无人机数量: {unconnected_uavs[0]}")
        
        # 保存视频
        if args.save_video and frames:
            with imageio.get_writer(video_path.replace(".mp4", f"_ep{episode+1}.mp4"), fps=10) as video:
                for frame in frames:
                    video.append_data(frame)
    
    # 打印总体统计信息
    print("\n测试结果汇总:")
    print(f"平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"平均覆盖率: {np.mean(all_coverage_rates)*100:.2f}% ± {np.std(all_coverage_rates)*100:.2f}%")
    print(f"平均最大覆盖率: {np.mean(all_max_coverage_rates)*100:.2f}% ± {np.std(all_max_coverage_rates)*100:.2f}%")
    print(f"平均未连接无人机数量: {np.mean(all_unconnected_uavs):.2f} ± {np.std(all_unconnected_uavs):.2f}")
    
    env.close()

if __name__ == "__main__":
    main() 