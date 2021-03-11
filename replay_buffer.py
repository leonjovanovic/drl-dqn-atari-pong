# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:03:09 2021

@author: Leon Jovanovic
"""
import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size, minimum, multi_step, gamma):
        self.size = size
        self.minimum = minimum
        # 'deque' is Doubly Ended Queuewhcih we use when we need quicker append and pop operations 
        # from both the ends of container - https://docs.python.org/2.5/lib/deque-objects.html    
        self.buffer = collections.deque(maxlen = size)
        # For multi_step we have to go multi_step number of transitions from one we decided to sample if it 
        # is possible (if its not done). After iterating, we need to remember last state, total rewards
        self.multi_step = multi_step
        # We will calculate each reward as reward*gamma^i
        self.gamma = gamma
        
    def append(self, transition):
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        chosen_transitions = random.sample(range(0, len(self.buffer) - 1), batch_size)
        
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        
        for transition in chosen_transitions:        
            i = 0
            total_reward = 0
            new_done = self.buffer[transition].done # Test and Delete!
            new_next_state = self.buffer[transition].next_state # Test and Delete!
            
            for i in range(self.multi_step):
                if transition + i < len(self.buffer):
                    total_reward += self.buffer[transition + i].reward * (self.gamma ** i)
                    new_done = self.buffer[transition + i].done
                    new_next_state = self.buffer[transition + i].next_state
                    # If we reached end of game dont look for more look ahead states
                    if self.buffer[transition + i].done:
                        i = self.multi_step

            states.append(self.buffer[transition].state)
            actions.append(self.buffer[transition].action)
            next_states.append(new_next_state)
            rewards.append(total_reward)
            dones.append(new_done)

        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))

