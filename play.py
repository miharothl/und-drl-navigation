import gym
# !pip3 install box2d
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

# !python -m pip install pyvirtualdisplay
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

from IPython import display
plt.ion()


env = gym.make('LunarLander-v2')
# env = gym.make('CarRacing-v0')
# env = gym.make('BipedalWalker-v3')
env = gym.make('CartPole-v1')

env.seed(0)
print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)
print('Number of actions: ', env.action_space)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=0)
# agent = Agent(state_size=24, action_size=4, seed=0)
agent = Agent(state_size=4, action_size=2, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(30):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()


