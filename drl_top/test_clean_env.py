# -*- coding: utf-8 -*-
"""
简化环境的测试脚本
测试训练好的MATD3模型在简化环境中的性能
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
from tqdm import tqdm
import time

# 导入简化环境和MATD3
import importlib.util
uav_env_path = os.path.join(parent_dir, 'uav_top_env', 'uav_env_clean.py')
spec = importlib.util.spec_from_file_location("uav_env_clean", uav_env_path)
uav_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uav_env_module)
UAVEnv = uav_env_module.UAVEnv

from matd3_no_gat import MATD3

# 获取当前文件目录路径
model_dir = os.path.join(current_dir, './output_clean_env/models/test1')  # 模型文件夹
video_dir = os.path.join(current_dir, './test_videos')  # 测试视频保存文件夹

# 创建目录
os.makedirs(video_dir, exist_ok=True)

def test_model(model_path, num_test_episodes=10, render=True, save_video=True):
    """测试训练好的模型"""
    print(f"🧪 开始测试模型: {model_path}")
    
    # 创建测试环境
    env = UAVEnv(
        render_mode='rgb_array' if save_video else ('human' if render else None),
        experiment_type='probabilistic',  # 测试概率驱动模式
        num_agents=6,
        num_targets=10,
        max_steps=200,
        min_active_agents=3,
        max_active_agents=6
    )
    
    print(f"✅ 测试环境创建成功")
    print(f"实验模式: {env.experiment_type}")
    
    # 获取环境信息
    obs, _ = env.reset()
    agents = env.agents
    
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]
    
    # 创建MATD3算法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}
    
    matd3 = MATD3(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=device
    )
    
    # 加载模型
    try:
        matd3.load(model_path)
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试统计
    test_results = {
        'episode_rewards': [],
        'episode_coverages': [],
        'episode_types': [],
        'topology_changes': [],
        'final_uav_counts': []
    }
    
    print(f"🎯 开始测试 ({num_test_episodes} episodes)...")
    
    for episode in tqdm(range(num_test_episodes), desc="测试进度"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_coverage_history = []
        
        # 记录episode信息
        episode_type = env.episode_plan['type']
        trigger_step = env.episode_plan['trigger_step']
        initial_uav_count = len(env.active_agents)
        
        # 视频录制
        frames = []
        record_video = save_video and (episode < 3)  # 只录制前3个episode
        
        print(f"\nEpisode {episode+1}: 类型={episode_type}" + 
              (f", 触发步数={trigger_step}" if trigger_step else ""))
        
        topology_changed = False
        change_step = None
        
        for step in range(env.max_steps):
            # 使用训练好的策略选择动作（无噪声）
            actions = matd3.select_action(obs, noise=0.0)
            
            # 记录变化前的UAV数量
            prev_uav_count = len(env.active_agents)
            
            # 执行动作
            next_obs, rewards, dones, truncated, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            
            # 检查拓扑变化
            current_uav_count = len(env.active_agents)
            if current_uav_count != prev_uav_count:
                topology_changed = True
                change_step = step + 1
                change_type = "损失" if current_uav_count < prev_uav_count else "添加"
                print(f"  第{step+1}步: {change_type} UAV ({prev_uav_count} → {current_uav_count})")
            
            # 计算覆盖率
            coverage_rate, _, _, _ = env.calculate_coverage_complete()
            episode_coverage_history.append(coverage_rate)
            
            obs = next_obs
            
            # 录制视频帧
            if record_video:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            elif render and not save_video:
                env.render(mode='human')
                time.sleep(0.05)  # 控制渲染速度
        
        # 记录episode结果
        final_coverage = np.mean(episode_coverage_history[-10:]) if episode_coverage_history else 0.0
        final_uav_count = len(env.active_agents)
        
        test_results['episode_rewards'].append(episode_reward)
        test_results['episode_coverages'].append(final_coverage)
        test_results['episode_types'].append(episode_type)
        test_results['topology_changes'].append(topology_changed)
        test_results['final_uav_counts'].append(final_uav_count)
        
        print(f"  结果: 奖励={episode_reward:.2f}, 覆盖率={final_coverage:.3f}, 最终UAV={final_uav_count}")
        
        # 保存视频
        if record_video and frames:
            video_path = f"{video_dir}/test_episode_{episode+1}_{episode_type}.mp4"
            save_video_file(frames, video_path)
            print(f"  📹 视频已保存: {video_path}")
    
    # 分析测试结果
    analyze_results(test_results)
    
    env.close()

def analyze_results(results):
    """分析测试结果"""
    print("\n" + "="*60)
    print("📊 测试结果分析")
    print("="*60)
    
    # 基础统计
    avg_reward = np.mean(results['episode_rewards'])
    std_reward = np.std(results['episode_rewards'])
    avg_coverage = np.mean(results['episode_coverages'])
    std_coverage = np.std(results['episode_coverages'])
    
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均覆盖率: {avg_coverage:.3f} ± {std_coverage:.3f}")
    
    # Episode类型分析
    episode_types = results['episode_types']
    normal_count = episode_types.count('normal')
    loss_count = episode_types.count('loss')
    addition_count = episode_types.count('addition')
    
    print(f"\nEpisode类型分布:")
    print(f"  正常: {normal_count} ({normal_count/len(episode_types)*100:.1f}%)")
    print(f"  损失: {loss_count} ({loss_count/len(episode_types)*100:.1f}%)")
    print(f"  添加: {addition_count} ({addition_count/len(episode_types)*100:.1f}%)")
    
    # 拓扑变化分析
    topology_changes = results['topology_changes']
    change_count = sum(topology_changes)
    print(f"\n拓扑变化执行:")
    print(f"  计划变化: {loss_count + addition_count}")
    print(f"  实际变化: {change_count}")
    print(f"  执行成功率: {change_count/(loss_count + addition_count)*100:.1f}%" if (loss_count + addition_count) > 0 else "  执行成功率: N/A")
    
    # UAV数量分析
    final_uav_counts = results['final_uav_counts']
    min_uavs = min(final_uav_counts)
    max_uavs = max(final_uav_counts)
    avg_uavs = np.mean(final_uav_counts)
    
    print(f"\nUAV数量统计:")
    print(f"  最少: {min_uavs}")
    print(f"  最多: {max_uavs}")
    print(f"  平均: {avg_uavs:.1f}")
    
    # 性能对比
    normal_rewards = [results['episode_rewards'][i] for i, t in enumerate(episode_types) if t == 'normal']
    change_rewards = [results['episode_rewards'][i] for i, t in enumerate(episode_types) if t != 'normal']
    
    if normal_rewards and change_rewards:
        print(f"\n性能对比:")
        print(f"  正常episode平均奖励: {np.mean(normal_rewards):.2f}")
        print(f"  变化episode平均奖励: {np.mean(change_rewards):.2f}")
        print(f"  性能差异: {np.mean(change_rewards) - np.mean(normal_rewards):.2f}")

def save_video_file(frames, path, fps=30):
    """保存视频文件"""
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

def test_random_policy():
    """测试随机策略作为基线"""
    print("\n🎲 测试随机策略基线...")
    
    env = UAVEnv(
        render_mode=None,
        experiment_type='probabilistic',
        num_agents=6,
        num_targets=10,
        max_steps=200
    )
    
    random_rewards = []
    random_coverages = []
    
    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # 随机动作
            actions = {agent: env.get_action_space(agent).sample() for agent in env.agents}
            obs, rewards, dones, _, _ = env.step(actions)
            episode_reward += sum(rewards.values())
        
        coverage_rate, _, _, _ = env.calculate_coverage_complete()
        random_rewards.append(episode_reward)
        random_coverages.append(coverage_rate)
    
    print(f"随机策略 - 平均奖励: {np.mean(random_rewards):.2f}, 平均覆盖率: {np.mean(random_coverages):.3f}")
    env.close()

def main():
    """主测试函数"""
    print("🧪 开始测试简化环境中的训练模型")

    # 模型路径
    model_path = f"{model_dir}/matd3_final.pth"

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 main_clean_env.py 进行训练")
        print(f"📁 期望模型路径: {model_path}")
        return
    
    # 测试训练好的模型
    test_model(
        model_path=model_path,
        num_test_episodes=20,
        render=False,  # 设置为True可以看到实时渲染
        save_video=True
    )
    
    # 测试随机策略基线
    test_random_policy()
    
    print("\n🎉 测试完成！")

if __name__ == '__main__':
    main()
