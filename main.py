# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:02:18 2021

@author: Leon Jovanovic
"""
import gym
from agent import Agent
import atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time
#---------------------------------Parameters----------------------------------

DQN_HYPERPARAMS = {
	'eps_start': 1,
    'eps_end': 0.02,
    'eps_decay': 10**5,
    'buffer_size':15000,
    'buffer_minimum':10001,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_nn': 1000,
    'multi_step': 1,
    'double_dqn': True
}

ENV_NAME = "PongNoFrameskip-v4"
RECORD = True
MAX_GAMES = 500
DEVICE = 'cuda'
BATCH_SIZE = 32

# For TensorBoard
SUMMARY_WRITER = True
LOG_DIR = 'content/runs'
name = '_'.join([str(k)+'.'+str(v) for k,v in DQN_HYPERPARAMS.items()])
name = 'prv'

#------------------------Create enviroment and agent--------------------------
env = atari_wrappers.make_env("PongNoFrameskip-v4")#gym.make("PongNoFrameskip-v4")
#For recording few seelcted episodes. 'force' means overwriting earlier recordings
if RECORD:
    env = gym.wrappers.Monitor(env, "main-"+ENV_NAME, force=True)
obs = env.reset()
#Create TensorBoard writer that will create graphs
writer = SummaryWriter(log_dir=LOG_DIR+'/'+name + str(time.time())) if SUMMARY_WRITER else None
#Create agent that will learn
agent = Agent(env, hyperparameters = DQN_HYPERPARAMS, device = DEVICE, writer = writer, max_games = MAX_GAMES)
#--------------------------------Learning-------------------------------------
num_games = 0
while num_games < MAX_GAMES:
    # Select one action with e-greedy policy and observe s,a,s',r and done
    action = agent.select_eps_greedy_action(obs)
    # Take that action and observe s, a, s', r and done
    new_obs, reward, done, _ = env.step(action)
    # Add s, a, s', r to buffer B
    agent.add_to_buffer(obs, action, new_obs, reward, done)
    # Sample a mini-batch from buffer B if B is large enough. If not skip until it is.
    # Use that mini-batch to improve NN value function approximation
    agent.sample_and_improve(BATCH_SIZE)
    
    obs = new_obs
    if(done):
        num_games = num_games + 1
        agent.print_info()
        agent.reset_parameters()
        obs = env.reset()
        
    
    
writer.close()    
gym.wrappers.Monitor.close(env)

#!tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Reinforcement Learning\reinforcement-learning-atari-pong\content\runs" --host=127.0.0.1