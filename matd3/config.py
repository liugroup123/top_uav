CONFIG = {
    "render_mode": "rgb_array",
    # "render_mode": "None",
    "num_episodes": 10000,
    "max_steps_per_episode": 200,
    "seed": 0,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "tau": 0.01,
    "batch_size": 64,
    "buffer_size": int(1e5),
    "policy_delay": 2,     # MATD3 特有
    "noise_std": 0.1,      # MATD3 特有
    "noise_clip": 0.5,     # MATD3 特有
    "initial_random_steps": 1000,
    "noise_decay_rate": 0.995,
}
