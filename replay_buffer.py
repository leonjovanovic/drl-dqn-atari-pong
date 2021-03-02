# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:03:09 2021

@author: Leon Jovanovic
"""
import collections
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, size, minimum):
        self.size = size
        self.minimum = minimum
        # 'deque' is Doubly Ended Queuewhcih we use when we need quicker append and pop operations 
        # from both the ends of container - https://docs.python.org/2.5/lib/deque-objects.html    
        self.buffer = collections.deque(maxlen = size)
        
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
            states.append(self.buffer[transition].state)
            actions.append(self.buffer[transition].action)
            next_states.append(self.buffer[transition].next_state)
            rewards.append(self.buffer[transition].reward)
            dones.append(self.buffer[transition].done)

        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))

        
        