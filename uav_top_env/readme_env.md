# 代码修改

## 环境代码
`uav_env_clean.py`
第一个版本的环境

`uav_env_clean_v2.py`
第二个版本的奖励函数，主要就是修改了奖励权重之类的

`uav_env_clean_v3.py`
第三个版本的环境，新增加了优先目标点这个东西对应的主训练代码就是`main_clean_env_v2.py`

>新增加了一个环境，主要就是修改平均距离这个奖励函数

`uav_env_clean_v4.py`—>`main_clean_env_v3.py`
1. 平均奖励
```python
r_s_d = (coverage_rate ** 2.0) * clipped_avg_min_distance
```
- 关键问题: 奖励与平均最小距离成正比。距离越大，奖励越大
- 后果: 鼓励UAV远离目标点，可能导致UAV移动到边界

2. 目标区别对待

原始的奖励就是不管对优先还是普通的目标点都是一样的，但是现在有先后顺序了


## 后续修改方向

1. 增加连通性的权重

2. 添加lstm到gat代码里面

3. 后续gat输出接一个attention

后续主要方向就是这个了

