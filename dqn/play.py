## watch an untrained agent
# state = env.reset()
# img = plt.imshow(env.render(mode='rgb_array'))
# for j in range(200):
#    action = agent.act(state)
#    img.set_data(env.render(mode='rgb_array'))
#    plt.axis('off')
#    display.display(plt.gcf())
#    display.clear_output(wait=True)
#    state, reward, done, _ = env.step(action)
#    if done:
#        break
#
# env.close()
import os
import pandas as pd

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
    def __init__(self, env, agent, model_id, path_models='models'):
        self.__env = env
        self.__agent = agent
        self.__model_id = model_id
        self.__path_models = path_models

    def play(self, score_max=True, score_med=False):

        filename = self.select_model_filename(score_max, score_med)

        self.__agent.qnetwork_local.load_state_dict(torch.load(filename))

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

    def select_model_filename(self, score_max=True, score_med=False):
        import re
        model_id = re.sub('[^0-9a-zA-Z.]+', '', self.__model_id)
        model_id = model_id.lower()

        import glob
        models = glob.glob(os.path.join(self.__path_models, "{}*".format(model_id)))

        models = [os.path.basename(m) for m in models]
        models = [m.rstrip(".pth") for m in models]
        models = [m.split('_') for m in models]

        df = pd.DataFrame(models,
                          columns=['model_id', 'timestamp', 'episode', 'score', 'epsilon'])

        df['score'] = df['score'].astype(float)

        select = ""
        if score_max:
            select = "Selected model (max score): {}"
            df = df.loc[df['score'] == df['score'].max()]
        elif score_med:
            select = "Selected model (median score): {}"
            df = df.loc[df['score'] == df['score'].median()]
        else:
            select = "Selected model (min score): {}"
            df = df.loc[df['score'] == df['score'].min()]

        timestamp = df['timestamp'].iloc[0]
        model_id = df['model_id'].iloc[0]
        episode = df['episode'].iloc[0]
        score = df['score'].iloc[0]
        epsilon = df['epsilon'].iloc[0]

        filename = "{}_{}_{}_{:.2f}_{}.pth".format(model_id, timestamp, episode, score, epsilon)

        path = os.path.join(self.__path_models, filename)

        print(select.format(path))

        return path
