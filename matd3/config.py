CONFIG = {
    "render_mode": "rgb_array",
    # "render_mode": "None",
    "num_episodes": 10000,
    "max_steps_per_episode": 200,
    "seed": 0,
    "actor_lr": 3e-4,        # 提高学习率以加速收敛
    "critic_lr": 3e-3,       # 提高学习率以加速收敛
    "gamma": 0.99,           # 提高折扣因子以更好地考虑长期回报
    "tau": 0.005,            # 降低软更新参数，提高稳定性
    "batch_size": 256,       # 增大批量大小以提高训练稳定性
    "buffer_size": int(5e5), # 增加缓冲区大小以存储更多经验
    "policy_delay": 2,       # MATD3 特有
    "noise_std": 0.2,        # 增加初始噪声以提高探索
    "noise_clip": 0.5,       # MATD3 特有
    "initial_random_steps": 5000,  # 增加初始随机步骤以填充缓冲区
    "noise_decay_rate": 0.9995,    # 缓慢衰减噪声
    "gradient_clip": 1.0,    # 梯度裁剪值
    "update_freq": 1,        # 每个环境步骤的更新频率
    "target_update_freq": 2, # 目标网络更新频率
}
