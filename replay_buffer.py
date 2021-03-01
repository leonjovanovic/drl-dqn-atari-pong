# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:03:09 2021

@author: Leon Jovanovic
"""
import collections

class ReplayBuffer():
    def __init__(self, size, minimum):
        self.size = size
        self.minimum = minimum
        # 'deque' is Doubly Ended Queuewhcih we use when we need quicker append and pop operations 
        # from both the ends of container - https://www.geeksforgeeks.org/deque-in-python/    
        self.buffer = collections.deque(maxlen = size)
        
    def append(self, transition):
        self.buffer.append(transition)