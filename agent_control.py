# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:27:50 2021

@author: Leon Jovanovic
"""
import torch
import numpy as np
from neural_nets import DQN

class AgentControl():
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        # We need to send both NNs to GPU hence '.to("cuda")
        self.moving_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        
    def select_greedy_action(self, obs):
        # We need to create tensor with data from obs. We need to transform obs to
        # numpy array because input to NN will be list with up to 32 obs (mini batches)
        # and this creates necessary format [1(up to 32),x,y,z] where x,y,z are tensor.shape
        # We need to send data to GPU hence '.to("cuda")
        tensor_obs = torch.tensor(np.array([obs])).to(self.device)
        all_actions = self.moving_nn(tensor_obs)
        # .max(1) returns tensor with value and tensor with its number (1 to 6), [1].item() returns only that number 
        return all_actions.max(1)[1].item()
        