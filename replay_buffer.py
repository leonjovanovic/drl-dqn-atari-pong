# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:03:09 2021

@author: Leon Jovanovic
"""
import collections
import random

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
        print(chosen_transitions)
        
        