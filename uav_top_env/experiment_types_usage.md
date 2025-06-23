# UAV拓扑环境实验类型使用指南

## 概述

UAV拓扑环境现在支持四种不同的实验类型，用于训练和测试不同场景下的多智能体协作策略：

1. **normal** - 正常模式（无拓扑变化）
2. **uav_loss** - UAV损失模式（只有UAV失效）
3. **uav_addition** - UAV添加模式（只有UAV添加）
4. **random_mixed** - 随机混合模式（随机失效或添加）

## 使用方法

### 基本初始化

```python
from uav_env_top import UAVEnv

# 创建不同实验类型的环境
env_normal = UAVEnv(experiment_type='normal')
env_loss = UAVEnv(experiment_type='uav_loss')
env_addition = UAVEnv(experiment_type='uav_addition')
env_mixed = UAVEnv(experiment_type='random_mixed')
```

### 自定义参数控制

现在支持精确控制拓扑变化的各种参数：

```python
# 自定义UAV损失模式
env_custom_loss = UAVEnv(
    experiment_type='uav_loss',
    topology_change_interval=30,    # 每30步失效一个UAV（默认80步）
    min_active_agents=2,            # 最少保持2个UAV（默认3个）
    max_active_agents=8             # 最多8个UAV（默认等于num_agents）
)

# 自定义UAV添加模式
env_custom_addition = UAVEnv(
    experiment_type='uav_addition',
    topology_change_interval=40,    # 每40步添加一个UAV（默认60步）
    initial_active_ratio=0.5,       # 从50%的UAV开始（默认约67%）
    max_active_agents=10            # 最多允许10个UAV
)

# 自定义随机混合模式
env_custom_mixed = UAVEnv(
    experiment_type='random_mixed',
    topology_change_probability=0.03, # 每步3%概率变化（默认1.5%）
    min_active_agents=1,             # 最少1个UAV
    max_active_agents=12             # 最多12个UAV
)
```

#### 可自定义的参数

- **topology_change_interval**: 固定间隔模式下的变化间隔（步数）
- **topology_change_probability**: 随机模式下每步的变化概率（0-1之间）
- **min_active_agents**: 最少保持的活跃UAV数量
- **max_active_agents**: 最多允许的活跃UAV数量
- **initial_active_ratio**: UAV添加模式下的初始活跃UAV比例（0-1之间）

### 实验类型详细说明

#### 1. Normal模式 (`experiment_type='normal'`)
- **特点**: 无拓扑变化，UAV数量保持恒定
- **用途**: 基础训练，建立稳定的协作策略
- **适用场景**: 理想环境下的多UAV协作

#### 2. UAV Loss模式 (`experiment_type='uav_loss'`)
- **特点**: 定期随机选择UAV失效
- **变化间隔**: 80步
- **最小UAV数**: 3个（保证基本功能）
- **用途**: 训练容错和重组能力
- **适用场景**: 模拟UAV故障、电池耗尽等情况

#### 3. UAV Addition模式 (`experiment_type='uav_addition'`)
- **特点**: 定期添加新的UAV
- **变化间隔**: 60步
- **最大UAV数**: 初始设定的num_agents
- **用途**: 训练动态扩展和集成能力
- **适用场景**: 模拟增援UAV到达、任务扩展等情况

#### 4. Random Mixed模式 (`experiment_type='random_mixed'`)
- **特点**: 随机进行UAV失效或添加
- **变化概率**: 每步1.5%的概率
- **用途**: 训练综合适应能力
- **适用场景**: 复杂动态环境

### 动态切换实验类型

```python
env = UAVEnv(experiment_type='normal')

# 运行一段时间后切换
env.set_experiment_type('uav_loss')

# 再次切换
env.set_experiment_type('random_mixed')
```

### 获取实验信息

```python
# 获取当前实验状态
info = env.get_experiment_info()
print(f"实验类型: {info['experiment_type']}")
print(f"活跃UAV数量: {info['active_agents_count']}")
print(f"拓扑变化进行中: {info['topology_in_progress']}")
```

## 训练策略建议

### 分阶段训练方案

1. **阶段1: 基础训练** (normal模式)
   - 训练稳定的协作策略
   - 建立基础的覆盖和连通性能力
   - 推荐训练轮数: 1000-2000轮

2. **阶段2: 容错训练** (uav_loss模式)
   - 学习UAV失效后的重组策略
   - 提高系统鲁棒性
   - 推荐训练轮数: 800-1500轮

3. **阶段3: 扩展训练** (uav_addition模式)
   - 学习集成新UAV的策略
   - 优化动态任务分配
   - 推荐训练轮数: 800-1500轮

4. **阶段4: 综合训练** (random_mixed模式)
   - 综合应对各种拓扑变化
   - 最终策略优化
   - 推荐训练轮数: 1000-2000轮

### 模型组合策略

```python
# 为不同实验类型训练专门的模型
models = {
    'normal': train_model(experiment_type='normal'),
    'loss': train_model(experiment_type='uav_loss'),
    'addition': train_model(experiment_type='uav_addition'),
    'mixed': train_model(experiment_type='random_mixed')
}

# 在综合实验中根据情况选择模型
def select_model(env_state):
    if env_state['topology_in_progress']:
        if env_state['change_type'] == 'failure':
            return models['loss']
        elif env_state['change_type'] == 'addition':
            return models['addition']
    return models['normal']
```

## 性能评估

### 关键指标

1. **覆盖率**: 目标点被覆盖的比例
2. **连通性**: UAV网络的连通性
3. **稳定性**: 性能指标的方差
4. **适应性**: 拓扑变化后的恢复速度

### 评估代码示例

```python
def evaluate_experiment_type(experiment_type, model):
    env = UAVEnv(experiment_type=experiment_type)
    
    coverage_rates = []
    recovery_times = []
    
    for episode in range(100):
        obs = env.reset()
        episode_coverage = []
        
        for step in range(500):
            actions = model.predict(obs)
            obs, rewards, dones, _, _ = env.step(actions)
            
            coverage_rate, _, _, _ = env.calculate_coverage_complete()
            episode_coverage.append(coverage_rate)
        
        coverage_rates.append(np.mean(episode_coverage))
    
    return {
        'avg_coverage': np.mean(coverage_rates),
        'coverage_std': np.std(coverage_rates)
    }
```

## 注意事项

1. **内存管理**: 不同实验类型可能有不同的内存使用模式
2. **训练时间**: 动态拓扑变化会增加训练复杂度
3. **超参数调整**: 不同实验类型可能需要不同的学习率和奖励权重
4. **模型保存**: 建议为每种实验类型保存独立的模型检查点

## 扩展功能

环境还提供了以下扩展功能：

- 自定义拓扑变化间隔
- 调整变化概率
- 设置最小/最大UAV数量限制
- 实时监控拓扑变化状态

这些功能可以通过修改`topology_config`参数来实现更精细的控制。
