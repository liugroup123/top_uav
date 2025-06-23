# UAV拓扑环境实验类型功能

## 🎯 功能概述

您的UAV拓扑环境现在支持四种不同的实验类型，可以模拟各种动态场景：

1. **normal** - 正常模式（无拓扑变化）
2. **uav_loss** - UAV损失模式（模拟UAV故障）
3. **uav_addition** - UAV添加模式（模拟增援到达）
4. **random_mixed** - 随机混合模式（随机损失或添加）

## 🚀 快速开始

### 基本使用

```python
from uav_env_top import UAVEnv

# 创建不同类型的实验环境
env = UAVEnv(experiment_type='uav_loss')  # UAV损失模式
obs, _ = env.reset()

for step in range(1000):
    actions = {f"agent_{i}": env.action_space[i].sample() for i in range(env.num_agents)}
    obs, rewards, dones, _, _ = env.step(actions)
    env.render()  # 可视化显示实验类型和UAV状态
```

### 自定义参数

```python
# 高度自定义的UAV损失实验
env = UAVEnv(
    num_agents=8,
    experiment_type='uav_loss',
    topology_change_interval=30,    # 每30步失效一个UAV
    min_active_agents=2,            # 最少保持2个UAV
    render_mode='human'
)

# 自定义UAV添加实验
env = UAVEnv(
    num_agents=6,
    experiment_type='uav_addition',
    initial_active_ratio=0.5,       # 从50%UAV开始
    topology_change_interval=40,    # 每40步添加一个
    max_active_agents=10            # 最多10个UAV
)

# 高频随机变化实验
env = UAVEnv(
    experiment_type='random_mixed',
    topology_change_probability=0.05, # 每步5%概率变化
    min_active_agents=1,
    max_active_agents=12
)
```

## 📊 参数说明

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `experiment_type` | 实验类型 | 'normal' | 'normal', 'uav_loss', 'uav_addition', 'random_mixed' |
| `topology_change_interval` | 变化间隔（步数） | 50-80 | 1+ |
| `topology_change_probability` | 随机变化概率 | 0.015-0.02 | 0-1 |
| `min_active_agents` | 最少UAV数量 | 3 | 1+ |
| `max_active_agents` | 最多UAV数量 | num_agents | 1+ |
| `initial_active_ratio` | 初始活跃比例 | 0.67 | 0-1 |

## 🔧 动态控制

```python
# 运行时切换实验类型
env.set_experiment_type('random_mixed')

# 动态调整参数
env.topology_config['change_interval'] = 20
env.topology_config['min_agents'] = 1

# 获取当前状态
info = env.get_experiment_info()
print(f"当前实验类型: {info['experiment_type']}")
print(f"活跃UAV数量: {info['active_agents_count']}")
```

## 📈 训练建议

### 分阶段训练策略

1. **基础训练** (normal模式) → 建立稳定协作
2. **容错训练** (uav_loss模式) → 学习故障应对
3. **扩展训练** (uav_addition模式) → 学习动态集成
4. **综合训练** (random_mixed模式) → 全面适应能力

### 模型组合策略

```python
# 为不同场景训练专门模型
models = {
    'normal': train_model(experiment_type='normal'),
    'loss': train_model(experiment_type='uav_loss'),
    'addition': train_model(experiment_type='uav_addition'),
    'mixed': train_model(experiment_type='random_mixed')
}

# 根据当前状态选择最适合的模型
def select_model(env_state):
    if env_state['topology_in_progress']:
        if env_state['change_type'] == 'failure':
            return models['loss']
        elif env_state['change_type'] == 'addition':
            return models['addition']
    return models['normal']
```

## 🎮 测试和演示

```bash
# 运行基本测试
python uav_env/simple_test.py

# 运行完整演示
python uav_env/test_experiment_types.py

# 运行自定义参数演示
python uav_env/custom_parameters_demo.py
```

## 📝 实验记录

环境会自动记录拓扑变化：

```python
# 获取变化历史
changes = []
for step in range(1000):
    prev_count = len(env.active_agents)
    env.step(actions)
    current_count = len(env.active_agents)
    
    if current_count != prev_count:
        changes.append({
            'step': step,
            'type': 'loss' if current_count < prev_count else 'addition',
            'from': prev_count,
            'to': current_count
        })

print(f"总共发生 {len(changes)} 次拓扑变化")
```

## 🎯 应用场景

- **UAV故障模拟**: 电池耗尽、硬件故障、通信中断
- **增援场景**: 新UAV加入、任务扩展、动态部署
- **复杂环境**: 战场环境、救援任务、动态监控
- **算法测试**: 鲁棒性测试、适应性评估、性能对比

## 🔍 可视化特性

- 实时显示当前实验类型
- 显示活跃/失效UAV状态
- 拓扑变化事件提示
- UAV数量动态统计

这个功能让您可以：
✅ 训练更鲁棒的多智能体系统
✅ 测试不同故障场景下的性能
✅ 研究动态拓扑下的协作策略
✅ 为实际部署做充分准备

现在您可以根据具体需求灵活调整所有参数，训练出适应各种动态环境的UAV协作模型！
