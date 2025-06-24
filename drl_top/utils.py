import numpy as np

# 噪声类
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.01, sigma=0.1, decay=0.999):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = max(0.01, sigma)  # 确保 sigma 在初始化时合理设置
        self.decay = decay
        self.reset()

    def reset(self, random_init=False):
        if random_init:
            self.state = np.random.randn(self.action_dim) * self.sigma
        else:
            self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def update_sigma(self):
        self.sigma = max(0.001, self.sigma * self.decay)  # 每个 episode 后减少噪声标准差
