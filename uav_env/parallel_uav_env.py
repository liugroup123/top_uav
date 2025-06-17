import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Tuple
from mpe_uav.uav_env.uav_env import UAVEnv

# 确保使用 spawn 方式
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

class SubprocEnv:
    """单个子进程环境"""
    def __init__(self, num_agents, num_targets, **kwargs):
        self.env = UAVEnv(num_agents=num_agents, num_targets=num_targets, **kwargs)
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def calculate_coverage_complete(self):
        return self.env.calculate_coverage_complete()
    
    def close(self):
        self.env.close()

def worker_process(remote, parent_remote, env_args):
    """工作进程函数"""
    parent_remote.close()
    env = SubprocEnv(**env_args)
    try:
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == 'step':
                    result = env.step(data)
                    remote.send(result)
                elif cmd == 'reset':
                    result = env.reset(**data)
                    remote.send(result)
                elif cmd == 'render':
                    result = env.render()
                    remote.send(result)
                elif cmd == 'coverage':
                    result = env.calculate_coverage_complete()
                    remote.send(result)
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
            except EOFError:
                break
    finally:
        env.close()

class ParallelUAVEnv:
    def __init__(self, num_envs, num_agents=5, num_targets=10, **kwargs):
        self.num_envs = num_envs
        self.num_agents = num_agents
        
        # 创建环境参数
        self.env_args = {
            'num_agents': num_agents,
            'num_targets': num_targets,
            **kwargs
        }
        
        # 创建临时环境获取空间信息
        temp_env = UAVEnv(**self.env_args)
        self.action_space = temp_env.action_space
        self.observation_space = temp_env.observation_space
        temp_env.close()
        
        # 创建进程和管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            process = mp.Process(
                target=worker_process,
                args=(work_remote, remote, self.env_args)
            )
            process.daemon = True
            process.start()
            work_remote.close()
            self.processes.append(process)
            
    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(('reset', kwargs))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)
    
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)
    
    def render(self):
        self.remotes[0].send(('render', None))
        return self.remotes[0].recv()
    
    def calculate_coverage_complete(self):
        self.remotes[0].send(('coverage', None))
        return self.remotes[0].recv()
    
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
            
    @property
    def agents(self):
        return [f"agent_{i}" for i in range(self.num_agents)]
    
    def get_action_space(self, agent_id):
        idx = int(agent_id.split('_')[1])
        return self.action_space[idx]
    
    def get_observation_space(self, agent_id):
        idx = int(agent_id.split('_')[1])
        return self.observation_space[idx]
