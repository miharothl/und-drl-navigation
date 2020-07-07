## watch an untrained agent
#state = env.reset()
#img = plt.imshow(env.render(mode='rgb_array'))
#for j in range(200):
#    action = agent.act(state)
#    img.set_data(env.render(mode='rgb_array'))
#    plt.axis('off')
#    display.display(plt.gcf())
#    display.clear_output(wait=True)
#    state, reward, done, _ = env.step(action)
#    if done:
#        break
#
#env.close()

import torch

import matplotlib.pyplot as plt
# %matplotlib inline

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

from IPython import display

class Play:
    def __init__(self, env, agent):
        self.__env = env
        self.__agent = agent

    def play(self):
         self.__agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

         for i in range(3):
             state = self.__env.reset()
             # img = plt.imshow(self.__env.render(mode='rgb_array'))
             self.__env.render(mode='rgb_array')
             for j in range(200):
                 action = self.__agent.act(state)
                 # img.set_data(self.__env.render(mode='rgb_array'))
                 self.__env.render(mode='rgb_array')
                 # plt.axis('off')
                 # display.display(plt.gcf())
                 # display.clear_output(wait=True)
                 state, reward, done, _ = self.__env.step(action)
                 if done:
                     break
         pass
