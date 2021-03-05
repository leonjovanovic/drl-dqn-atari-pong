# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:27:50 2021

@author: Leon Jovanovic
"""
import torch
import numpy as np
from neural_nets import DQN
import torch.optim as optim
import torch.nn as nn
from apex import amp

class AgentControl():
    
    def __init__(self, env, device, lr, gamma, multi_step, double_dqn):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.multi_step = multi_step
        self.double_dqn = double_dqn
        # We need to send both NNs to GPU hence '.to("cuda")
        self.moving_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn = DQN(input_shape = env.observation_space.shape, num_of_actions = env.action_space.n).to(device)
        self.target_nn.load_state_dict(self.moving_nn.state_dict())
        self.optimizer = optim.Adam(self.moving_nn.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        '''
        self.moving_nn, self.optimizer = amp.initialize(
           self.moving_nn, self.optimizer, opt_level="O3"
        )'''
        
    def select_greedy_action(self, obs):
        # We need to create tensor with data from obs. We need to transform obs to
        # numpy array because input to NN will be list with up to 32 obs (mini batches)
        # and this creates necessary format [1(up to 32),x,y,z] where x,y,z are tensor.shape
        # We need to send data to GPU hence '.to("cuda")
        tensor_obs = torch.tensor(np.array([obs])).to(self.device)
        all_actions = self.moving_nn(tensor_obs)
        # .max(1) returns tensor with value and tensor with its number (1 to 6), [1].item() returns only that number 
        return all_actions.max(1)[1].item()
    
    def improve(self, mini_batch):
        # Calculate loss
        loss = self.calc_loss(mini_batch)
        # Improve NN parameters
        self.improve_params(loss)
        return loss.item()
        
    def calc_loss(self, mini_batch):
        states, actions, next_states, rewards, dones = mini_batch
        # Transform numpy array to Tensor and send it to GPU
        states_tensor = torch.as_tensor(states).to(self.device)
        next_states_tensor = torch.as_tensor(next_states).to(self.device)
        actions_tensor = torch.as_tensor(actions).to(self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.uint8).to(self.device)
        
        # First we need to find value of action we decided to do
        # From inputing states into NN, we will get output matrix BATCH_SIZEx6
        # Then with tensor.gather(dimension, index) we find that value with index from actions
        # Finally we use squeeze(-1) to reduce dimensions from 2 to 1
        curr_state_action_value = self.moving_nn(states_tensor).gather(1,actions_tensor[:,None]).squeeze(-1)
        
        if self.double_dqn:
            # Double Q Learning will be implemented with getting max action (serial number) from first NN for each of 32 states,
            # then we get 32x6 output of second NN and we take value from 32x6 matrix that is on place of serial number from first
            # Take best action's serial number from first NN
            double_dqn_max_action = self.moving_nn(next_states_tensor).max(1)[1]
            double_dqn_max_action.detach()
            # Get 1x6 with action values from second NN
            second_nn_actions = self.target_nn(next_states_tensor)
            next_state_action_value = second_nn_actions.gather(1, double_dqn_max_action[:,None]).squeeze(-1)
        else:
            # We need to find best next action and we will get that by appling NN with older params (target_NN)
            # which we will update to new after X iteration. Using old params we avoid Q Learning problem with not
            # converging since if we use only one NN its not gradient descent.
            next_state_action_value = self.target_nn(next_states_tensor).max(1)[0]
        # We do differentiation for moving_nn (w or curr_state_action_value) and we dont do it for target_nn (w'),
		# so we dont have to remember operations for backprop. Good for huge amount of operations
        next_state_action_value = next_state_action_value.detach()
        # Calculate Q-target
        Q_target = rewards_tensor + (self.gamma ** self.multi_step)* next_state_action_value
        # Apply MSE Loss which will be applied to all BATCH_SIZEx1 rows and output will be 1x1 
        return self.loss(curr_state_action_value, Q_target)
        
    def improve_params(self, loss):
        # Reset the grads
        self.optimizer.zero_grad()
		# Do backpropagation, backward calculates derivative over w since we detached w'
        # Derivative values are stored inside moving_nn parameters! Since optimizer
        # is created over moving_nn he will know where to look for derivative values
        loss.backward()
        #with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #    scaled_loss.backward()
        # One step of optimization. We optimize w and will update that w' = w after iter_update_target iterations
        # .step() calculates w = w - gradient*loss. W, gradient and loss are stored inside moving_nn parameters
        self.optimizer.step()
        
    def update_target_nn(self):
        self.target_nn.load_state_dict(self.moving_nn.state_dict())
        
        
    
        