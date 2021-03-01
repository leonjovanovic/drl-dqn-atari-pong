# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:17:45 2021

@author: Leon Jovanovic
"""
import torch.nn as nn
import torch
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super(DQN, self).__init__()
        # We need Convolution NN to analyze input picture from current frame
        self.conv_nn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU() 
        )
        # Calculation of output of CNN so we can tell rest of NN what to expect on input.
        # 'input_shape' had to be 1 dimension lower because we dont know size of that dim upfornt
        # So we need to add it every time if we want single frame to run through CNN
        # np.prod flattens output by product of sizes of every dimension
        cnn_output_shape = self.conv_nn(torch.zeros(1, *input_shape))
        cnn_output_shape = int(np.prod(cnn_output_shape.size()))
        # Output of regular NN will be 1x6 where 6 stands for 6 actions and how much NN thinks each action is right one
        self.linear_nn = nn.Sequential(
			nn.Linear(cnn_output_shape, 512),
			nn.ReLU(),
			nn.Linear(512, num_of_actions)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        batch_size = x.size()[0] # Bacth size will be either 1 or BATCH_SIZE
        # We need to flatten result of CNN and 'view' reshapes tensor to have 'batch_size' rows and data/batch_size columns (that is -1)
        cnn_output = self.conv_nn(x).view(batch_size, -1)        
        return self.linear_nn(cnn_output) # apply rest of NN
