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

# watch an untrained agent
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


#
# # load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
#
# for i in range(30):
#     state = env.reset()
#     img = plt.imshow(env.render(mode='rgb_array'))
#     for j in range(200):
#         action = agent.act(state)
#         img.set_data(env.render(mode='rgb_array'))
#         plt.axis('off')
#         display.display(plt.gcf())
#         display.clear_output(wait=True)
#         state, reward, done, _ = env.step(action)
#         if done:
#             break
#
# env.close()


def plot(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def dqn(n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.997):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Eps: {}'.format(i_episode, np.mean(scores_window), eps))
            plot(scores)
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

        if np.mean(scores_window)>=300.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    return scores

scores = dqn()







