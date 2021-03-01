# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:15:15 2021

@author: Leon Jovanovic
"""
from numpy import random
from agent_control import AgentControl
from replay_buffer import ReplayBuffer
from collections import namedtuple


class Agent():
    
    Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'), rename = False) # 'rename' means not to overwrite invalid field
    
    def __init__(self, env, hyperparameters, device):
        self.eps_start = hyperparameters['eps_start']
        self.eps_end = hyperparameters['eps_end']
        self.eps_decay = hyperparameters['eps_decay']
        self.epsilon = hyperparameters['eps_start']
        
        self.env = env
        
        self.agent_control = AgentControl(env, device)
        self.replay_buffer = ReplayBuffer(hyperparameters['buffer_size'], hyperparameters['buffer_minimum'])
        self.num_iterations = 0
        
    def select_greedy_action(self, obs):
        # Give current state to the control who will pass it to NN which will
        # return all actions and the control will take max and return it here
        return self.agent_control.select_greedy_action(obs)
        
    def select_eps_greedy_action(self, obs):
        rand_num = random.rand()
        if self.epsilon > rand_num:
            # Select random action - explore
            return self.env.action_space.sample()
        else:
            # Select best action
            return self.select_greedy_action(obs)
        
    def add_to_buffer(self, obs, action, new_obs, reward, done):
        transition = self.Transition(state = obs, action = action, next_state = new_obs, reward = reward, done = done)
        self.replay_buffer.append(transition)
        self.num_iterations = self.num_iterations + 1
        if self.epsilon > self.eps_end:
            self.epsilon = self.eps_start - self.num_iterations / self.eps_decay
        
    def sample_and_improve(self):
        
            
            
        
        