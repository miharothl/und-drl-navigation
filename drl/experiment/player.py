import os

import pandas as pd
import torch
import matplotlib.pyplot as plt
from numpy import random

from drl.experiment.config import Config
from drl.experiment.recorder import Recorder
from drl.image import imshow


class Player:
    def __init__(self, env, agent, model_id, config: Config, session_id, path_models='models'):
        self.__env = env
        self.__agent = agent
        self.__model_id = model_id
        self.__config = config
        self.__session_id = session_id
        self.__path_models = path_models

    def play(self, trained, mode, is_rgb, model_filename, num_episodes, num_steps):

        if is_rgb:
            return self.play_rgb(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode, model_filename=model_filename)
        else:
            return self.play_classic(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode, model_filename=model_filename)

    def play_classic( self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array', model_filename=None, num_steps=None):

        recorder = Recorder(header=['episode', 'step', 'action', 'reward', 'reward_total'],
                            session_id=self.__session_id,
                            experiments_path=self.__config.get_app_experiments_path(train_mode=False),
                            model=None)

        if trained or (model_filename is not None):
            filename = self.select_model_filename(score_max, score_med, model_filename=model_filename)
            self.__agent.qnetwork_local.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

        for i in range(num_episodes):
            scores = []
            reward_total = 0
            step = 0
            state = self.__env.reset()
            self.__env.render(mode=mode)

            lives = -1
            new_life = False

            if num_steps is None:
                while True:
                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)
                    # action= random.choice([1,2,3])

                    if new_life:
                        action = 0

                    action = action + 1
                    self.__env.render(mode=mode)

                    state, reward, done, info = self.__env.step(action)

                    if info['ale.lives'] > lives:
                        lives = info['ale.lives']
                        new_life = True
                    elif info['ale.lives'] < lives:
                        lives = info['ale.lives']
                        new_life = True
                        reward = reward - 1
                    else:
                        new_life = False

                    reward_total += reward
                    step += 1

                    recorder.record([i, step, action, reward, reward_total])

                    if done:
                        break
            else:
                for j in range(num_steps):
                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    step += 1

                    recorder.record([i, step, action, reward, reward_total])

                    if done:
                        break

            scores.append([[i, reward_total, step]])

            recorder.save()

        return scores

    def play_rgb(self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array', model_filename=None, num_steps=None):

        if trained or (model_filename is not None):
            filename = self.select_model_filename(score_max, score_med, model_filename=model_filename)
            self.__agent.qnetwork_local.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

        for i in range(num_episodes):
            scores = []
            reward_total = 0
            steps_total = 0
            state = self.__env.reset()
            image = self.__env.render(mode=mode)

            image2 = imshow(state)

            if num_steps is None:
                while True:
                    action = self.__agent.act(image2)
                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    steps_total += 1

                    if done:
                        break
            else:
                for j in range(num_steps):
                    action = self.__agent.act(image2)
                    self.__env.render(mode=mode)
                    state, reward, done, _ = self.__env.step(action)
                    reward_total += reward
                    steps_total += j

                    if done:
                        break
            scores.append([[i, reward_total, steps_total]])

        return scores

    def play_plot(self, score_max=True, score_med=False):

        filename = self.select_model_filename(score_max, score_med)

        self.__agent.qnetwork_local.load_state_dict(torch.load(filename))

        # from pyvirtualdisplay import Display
        # display = Display(visible=0, size=(1400, 900))
        # display.start()

        # is_ipython = 'inline' in plt.get_backend()
        # if is_ipython:
        #     from IPython import display
        #

        from IPython import display

        for i in range(1):
            state = self.__env.reset()
            img = plt.imshow(self.__env.render(mode='rgb_array'))
            for j in range(200):
                action = self.__agent.act(state)
                img.set_data(self.__env.render(mode='rgb_array'))
                plt.axis('off')
                display.display(plt.gcf())
                display.clear_output(wait=True)
                state, reward, done, _ = self.__env.step(action)
                if done:
                    break

        # display.stop()

    def play_banana(self, score_max=True, score_med=False):

        filename = self.select_model_filename(score_max, score_med)

        self.__agent.qnetwork_local.load_state_dict(torch.load(filename))


        brain_name = self.__env.brain_names[0]
        brain = self.__env.brains[brain_name]

        # reset the environment
        env_info = self.__env.reset(train_mode=True)[brain_name]

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))

        # number of actions
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

        env_info = self.__env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = self.__agent.act(state)
            env_info = self.__env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                print('done')
                break

        print("Score: {}".format(score))


    def get_model_filename(self, model_filename):
        path = os.path.join(self.__path_models, model_filename)
        return path

    def select_model_filename(self, score_max=True, score_med=False, model_filename=None):
        if model_filename is not None:
            path = os.path.join(self.__path_models, model_filename)
            return path

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
