# -*- coding: utf-8 -*-
"""
LSTM增强的MATD3测试脚本 - 测试训练好的LSTM-GAT模型
基于原有测试代码，集成LSTM时序建模能力
"""

import sys
import os
import torch
import numpy as np
import time

# 获取当前文件的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入环境和算法
import importlib.util

# 导入原有环境
uav_env_path = os.path.join(parent_dir, 'uav_top_env', 'uav_env_clean.py')
spec = importlib.util.spec_from_file_location("uav_env_clean", uav_env_path)
uav_env_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uav_env_module)
UAVEnv = uav_env_module.UAVEnv

# 导入LSTM GAT模型
gat_lstm_path = os.path.join(parent_dir, 'uav_top_env', 'gat_model_lstm.py')
spec = importlib.util.spec_from_file_location("gat_model_lstm", gat_lstm_path)
gat_lstm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gat_lstm_module)
UAVAttentionNetworkLSTM = gat_lstm_module.UAVAttentionNetworkLSTM

from matd3_no_gat import MATD3

# LSTM模型路径
model_dir = os.path.join(current_dir, './output_clean_env_lstm/models/test1')

class UAVEnvLSTM(UAVEnv):
    """LSTM增强的UAV环境 - 最小修改版本"""

    def __init__(self, enable_temporal=True, temporal_steps=5, **kwargs):
        # 保存LSTM配置
        self.enable_temporal = enable_temporal
        self.temporal_steps = temporal_steps

        # 调用父类初始化
        super().__init__(**kwargs)

        # 替换GAT模型为LSTM版本
        if self.enable_temporal:
            self._init_lstm_gat_model()

    def _init_lstm_gat_model(self):
        """初始化LSTM增强的GAT模型"""
        self.gat_model = UAVAttentionNetworkLSTM(
            uav_features=self.uav_features,
            target_features=self.target_features,
            hidden_size=self.gat_hidden_size,
            heads=self.gat_heads,
            dropout=self.gat_dropout,
            output_dim=self.gat_output_dim,
            enable_temporal=self.enable_temporal,
            temporal_steps=self.temporal_steps,
            device=self.device
        )
        self.gat_model.train()

    def reset(self, seed=None, options=None):
        """重置环境，同时重置LSTM状态"""
        if hasattr(self.gat_model, 'reset_temporal_state'):
            self.gat_model.reset_temporal_state()
        return super().reset(seed=seed, options=options)

def test_model(model_path, num_test_episodes=5, render_mode='human', test_mode='mixed'):
    """
    简化的模型测试函数
    test_mode: 'normal', 'loss', 'addition', 'mixed'
    """
    print(f"🧪 测试模型: {model_path}")
    print(f"🎯 测试模式: {test_mode}")

    # 创建LSTM增强测试环境 (与训练环境一致)
    env = UAVEnvLSTM(
        render_mode=render_mode,
        experiment_type='probabilistic',
        num_agents=5,        # 与训练一致
        num_targets=10,
        max_steps=500,       # 与训练一致
        min_active_agents=4, # 与训练一致
        max_active_agents=6,
        # LSTM时序配置
        enable_temporal=True,
        temporal_steps=5
    )

    print(f"✅ LSTM增强环境创建成功 | 模式: {render_mode}")
    print(f"🧠 GAT架构: 双GAT (UAV-UAV + UAV-Target)")
    print(f"📊 GAT配置: {env.gat_hidden_size}隐藏层, {env.gat_heads}头, 输出{env.gat_output_dim}维")

    # LSTM时序信息显示
    if env.enable_temporal:
        print(f"⏰ 时序LSTM: 启用 ({env.temporal_steps}步历史)")
    else:
        print(f"⏰ 时序LSTM: 禁用")

    if env.use_obs_attention:
        print(f"🔗 观察Attention: 启用 ({env.obs_attention_hops}跳, {env.obs_attention_heads}头)")
    else:
        print(f"🔗 观察Attention: 禁用")

    # 获取环境信息
    obs, _ = env.reset()
    agents = env.agents
    obs_dim = env.get_observation_space(agents[0]).shape[0]
    action_dim = env.get_action_space(agents[0]).shape[0]

    print(f"📊 观察空间维度: {obs_dim} (含{env.gat_output_dim}维GAT特征)")

    # 创建并加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matd3 = MATD3(
        agents=agents,
        obs_dims={agent: obs_dim for agent in agents},
        action_dims={agent: action_dim for agent in agents},
        device=device
    )

    try:
        matd3.load(model_path)
        print(f"✅ MATD3模型加载成功")

        # 加载对应的LSTM-GAT模型
        gat_path = model_path.replace('matd3_', 'gat_')
        if os.path.exists(gat_path):
            env.load_gat_model(gat_path)
            print(f"✅ LSTM-GAT+观察Attention模型加载成功: {gat_path}")

            # 验证GAT模型结构
            gat_layers = list(env.gat_model.model.keys())
            uav_gat_layers = [k for k in gat_layers if 'uav_gat' in k]
            target_gat_layers = [k for k in gat_layers if 'uav_target_gat' in k]
            lstm_layers = [k for k in gat_layers if 'temporal' in k]

            print(f"📊 UAV-UAV GAT层: {len(uav_gat_layers)}")
            print(f"📊 UAV-Target GAT层: {len(target_gat_layers)}")
            print(f"📊 LSTM时序层: {len(lstm_layers)}")
            print(f"📊 GAT输出维度: {env.gat_output_dim}")

            # 验证观察Attention结构
            if env.use_obs_attention and env.obs_attention_net is not None:
                obs_attention_layers = list(env.obs_attention_net.attention_layers)
                print(f"📊 观察Attention层: {len(obs_attention_layers)}")
                print(f"📊 Attention跳数: {env.obs_attention_net.num_hops}")
            else:
                print(f"📊 观察Attention: 未启用")
        else:
            print(f"⚠️  GAT模型文件不存在: {gat_path}")
            print("⚠️  将使用随机初始化的双GAT+观察Attention (可能影响性能)")
            print("💡 建议使用训练好的完整模型以获得最佳效果")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 测试统计
    episode_rewards = []
    episode_coverages = []

    print(f"🎯 开始测试 ({num_test_episodes} episodes)...")

    for episode in range(num_test_episodes):
        obs, _ = env.reset()

        # 强制设置episode模式
        if test_mode != 'mixed':
            env._force_episode_mode(test_mode)

        episode_reward = 0
        coverage_history = []

        # 获取episode信息
        episode_type = env.episode_plan['type']
        trigger_step = env.episode_plan['trigger_step']

        print(f"\nEpisode {episode+1}: {episode_type}" +
              (f" (触发步数: {trigger_step})" if trigger_step else ""))

        for step in range(env.max_steps):
            # 使用训练好的策略 (无噪声)
            actions = matd3.select_action(obs, noise=0.0)

            # 执行动作
            obs, rewards, dones, _, _ = env.step(actions)
            episode_reward += sum(rewards.values())

            # 记录覆盖率
            coverage_rate, _, _, _ = env.calculate_coverage_complete()
            coverage_history.append(coverage_rate)

            # 渲染 (如果是human模式)
            if render_mode == 'human':
                env.render()
                time.sleep(0.02)  # 控制播放速度

        # 计算episode结果
        final_coverage = np.mean(coverage_history[-10:]) if coverage_history else 0.0
        max_coverage = max(coverage_history) if coverage_history else 0.0
        final_uav_count = len(env.active_agents)

        # 记录统计
        episode_rewards.append(episode_reward)
        episode_coverages.append(final_coverage)

        print(f"  奖励: {episode_reward:.1f} | 覆盖率: {final_coverage:.3f} | 最大: {max_coverage:.3f} | UAV: {final_uav_count}")

    # 简单统计
    print(f"\n📊 测试结果:")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均覆盖率: {np.mean(episode_coverages):.3f}")
    print(f"最高覆盖率: {max(episode_coverages):.3f}")

    # LSTM增强架构验证
    print(f"\n🧠 LSTM增强架构验证:")
    print(f"✅ 双GAT架构 (UAV-UAV + UAV-Target)")
    print(f"✅ GAT特征维度: {env.gat_output_dim}")

    if env.enable_temporal:
        print(f"✅ 时序LSTM ({env.temporal_steps}步历史)")
        if env.use_obs_attention:
            print(f"✅ 观察Attention ({env.obs_attention_hops}跳, {env.obs_attention_heads}头)")
            print(f"✅ 完整信息处理流程: 基础观察 → GAT → LSTM时序 → 观察Attention → RL算法")
        else:
            print(f"✅ 观察Attention: 禁用")
            print(f"✅ 信息处理流程: 基础观察 → GAT → LSTM时序 → RL算法")
    else:
        print(f"✅ 时序LSTM: 禁用")
        if env.use_obs_attention:
            print(f"✅ 观察Attention ({env.obs_attention_hops}跳, {env.obs_attention_heads}头)")
            print(f"✅ 信息处理流程: 基础观察 → GAT → 观察Attention → RL算法")
        else:
            print(f"✅ 观察Attention: 禁用")
            print(f"✅ 信息处理流程: 基础观察 → GAT → RL算法")

    print(f"✅ 观察空间总维度: {obs_dim}")

    env.close()



def main():
    """简化的主测试函数"""
    print("🧪 测试训练好的模型")

    # 模型路径 (可以修改为具体的模型文件)
    model_path = f"{model_dir}/matd3_final.pth"

    # 如果没有final模型，尝试最新的模型
    if not os.path.exists(model_path):
        # 查找最新的MATD3模型文件
        model_files = [f for f in os.listdir(model_dir) if f.startswith('matd3_episode_') and f.endswith('.pth')]
        if model_files:
            # 按episode数排序，取最新的
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(model_dir, model_files[-1])
            print(f"📁 使用最新MATD3模型: {model_files[-1]}")

            # 检查对应的LSTM-GAT模型是否存在
            gat_file = model_files[-1].replace('matd3_', 'gat_')
            gat_path = os.path.join(model_dir, gat_file)
            if os.path.exists(gat_path):
                print(f"📁 找到对应LSTM-GAT模型: {gat_file} (双GAT+LSTM架构)")
            else:
                print(f"⚠️  未找到对应LSTM-GAT模型: {gat_file}")
                print("💡 将使用随机初始化的LSTM-GAT，建议重新训练")
        else:
            print(f"❌ 未找到模型文件在: {model_dir}")
            print("请先运行训练代码")
            return

    # 测试不同模式
    print("\n� 可选测试模式:")
    print("1. mixed    - 混合模式 (随机选择)")
    print("2. normal   - 正常模式 (无拓扑变化)")
    print("3. loss     - 损失模式 (UAV失效)")
    print("4. addition - 增加模式 (UAV增加)")

    # 可以在这里修改测试模式
    test_model(
        model_path=model_path,
        num_test_episodes=5,
        render_mode='human',  # 改为 'rgb_array' 可以录制视频
        test_mode = 'mixed'  # 改为 'normal', 'loss', 'addition' 测试特定模式
    )

    print("\n🎉 测试完成！")

if __name__ == '__main__':
    main()
