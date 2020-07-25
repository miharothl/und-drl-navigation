import os
from datetime import datetime
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import sys

from drl.experiment.config import Config
from drl.experiment.recorder import Recorder


class Trainer:
    def __init__(self, model_id, config: Config, session_id, path_models='models'):
        self.__model_id = model_id
        self.__timestamp = self.get_timestamp()
        self.__config = config
        self.__session_id = session_id
        self.__path_models = path_models

    def plot(self, scores, filename):
        # plot the scores
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(filename)
        plt.close()

    def get_timestamp(self):
        # return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return datetime.now().strftime("%Y%m%dT%H%M")

    def get_model_filename(self, episode, score, eps):
        import re
        model_id = re.sub('[^0-9a-zA-Z]+', '', self.__model_id)
        model_id = model_id.lower()
        filename = "{}_{}_{}_{:.2f}_{:.2f}.pth".format(model_id, self.__timestamp, episode, score, eps)
        return os.path.join(self.__path_models, filename)

    def get_plot_filename(self, episode, score, eps):
        import re
        model_id = re.sub('[^0-9a-zA-Z]+', '', self.__model_id)
        model_id = model_id.lower()
        filename = "{}_{}_{}_{:.2f}_{:.2f}.jpg".format(model_id, self.__timestamp, episode, score, eps)
        return os.path.join(self.__path_models, filename)

    def train(self, agent, env, is_rgb, model_filename=None, num_episodes=10000):

        if is_rgb:
            return self.dqn_rgb(agent, env, model_filename, n_episodes=num_episodes)
        else:
            return self.dqn_normal(agent, env, model_filename, n_episodes=num_episodes)

    def select_model_filename(self, model_filename=None):
        if model_filename is not None:
            path = os.path.join(self.__path_models, model_filename)
            return path

    def dqn_normal(self, agent, env, model_filename=None, n_episodes=10000, max_t=5000, eps_start=1.0, eps_end=0.15,
                   eps_decay=0.9990, terminate_soore=800.0):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon

        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.qnetwork_local.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.qnetwork_target.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            eps = 0.78

        recorder = Recorder(header=['episode', 'step', 'action', 'reward', 'reward_total'],
                            session_id=self.__session_id,
                            experiments_path=self.__config.get_app_experiments_path(train_mode=True),
                            model=None)

        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            score = 0

            if self.__config.get_current_env_is_atari_flag():
                lives = -1
                new_life = False

            for t in range(max_t):
                action = agent.act(state, eps)

                if self.__config.get_current_env_is_atari_flag():
                    if self.__config.get_current_agent_start_game_action_required():
                        if new_life:
                            action = self.__config.get_current_agent_start_game_action()

                action = action + self.__config.get_current_agent_state_offset()

                if 0: #debug
                    env.render(mode='human')

                next_state, reward, done, info = env.step(action)

                # debug
                if reward > 0:
                    print('\nreward!')

                if self.__config.get_current_env_is_atari_flag():
                    if info['ale.lives'] > lives:
                        lives = info['ale.lives']
                        new_life = True
                    elif info['ale.lives'] < lives:
                        lives = info['ale.lives']
                        new_life = True
                        reward = reward - 1
                    else:
                        new_life = False

                recorder.record([i_episode, t, action, reward, score])

                action = action - self.__config.get_current_agent_state_offset()

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score

            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            model_filename = self.get_model_filename(i_episode, np.mean(scores_window), eps)
            
            sys.stdout.flush()

            print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
                  end="")
            if i_episode % 100 == 0:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                    eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)

            if i_episode % 100 == 0:
                plot_filename = self.get_plot_filename(i_episode, np.mean(scores_window), eps)
                # self.plot(scores, plot_filename)

            if i_episode % 20 == 0:
                agent.check_memory()

            if np.mean(scores_window) >= terminate_soore:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode - 100,
                                                                                    np.mean(scores_window), eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)
                self.plot(scores)
                break

            recorder.save()

        return scores

    def dqn_rgb(self, agent, env, model_filename=None, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.05,
                eps_decay=0.995, terminate_soore=800.0):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon

        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.qnetwork_local.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.qnetwork_target.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            eps = 0.78

        for i_episode in range(1, n_episodes + 1):
            state = env.reset()

            from drl.image import imshow
            state = imshow(state)

            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)

                from drl.image import imshow
                next_state = imshow(next_state)

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    agent.check_memory()
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score

            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            model_filename = self.get_model_filename(i_episode, np.mean(scores_window), eps)

            sys.stdout.flush()

            print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
                  end="")
            if i_episode % 100 == 0:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                    eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)

            if i_episode % 100 == 0:
                plot_filename = self.get_plot_filename(i_episode, np.mean(scores_window), eps)
                # self.plot(scores, plot_filename)

            if np.mean(scores_window) >= terminate_soore:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode - 100,
                                                                                    np.mean(scores_window), eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)
                self.plot(scores)
                break

        return scores

    def dqn_banana(self, agent, env, n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                   terminate_soore=300.0):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):

            # reset the environment
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]  # get the current state
            score = 0

            for t in range(max_t):

                action = agent.act(state, eps)

                env_info = env.step(action)[brain_name]  # send the action to the environment

                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished

                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    print('done')
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score

            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            model_filename = self.get_model_filename(i_episode, np.mean(scores_window), eps)

            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
                  end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                    eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)

            if i_episode % 500 == 0:
                self.plot(scores)

            if np.mean(scores_window) >= terminate_soore:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode - 100,
                                                                                    np.mean(scores_window), eps))
                torch.save(agent.qnetwork_local.state_dict(), model_filename)
                self.plot(scores)
                break

        return scores
