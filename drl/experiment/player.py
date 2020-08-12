import torch
import matplotlib.pyplot as plt

from drl.environments.environment import Environment
from drl.experiment.config import Config
from drl.experiment.recorder import Recorder
from drl.image import imshow


class Player:
    def __init__(self, env: Environment, agent, model_id, config: Config, session_id, path_models='models'):
        self.__env = env
        self.__agent = agent
        self.__model_id = model_id
        self.__config = config
        self.__session_id = session_id
        self.__path_models = path_models

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__env.close()

    def play(self, trained, mode, is_rgb, model_filename, num_episodes, num_steps):

        if is_rgb:
            return self.play_rgb(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode,
                                 model_filename=model_filename)
        else:
            return self.play_classic(num_episodes=num_episodes, num_steps=num_steps, trained=trained, mode=mode,
                                     model_filename=model_filename)

    def play_classic(self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array',
                     model_filename=None, num_steps=None):

        recorder = Recorder(header=['episode', 'step', 'action', 'reward', 'reward_total'],
                            session_id=self.__session_id,
                            experiments_path=self.__config.get_app_experiments_path(train_mode=False),
                            model=None)

        if trained or (model_filename is not None):
            self.__agent.current_model.load_state_dict(
                torch.load(model_filename, map_location=lambda storage, loc: storage))

        if mode == "human":
            train_mode = False
        else:
            train_mode = True

        scores = []
        for i in range(num_episodes):
            reward_total = 0
            step = 0
            state, new_life = self.__env.reset(train_mode=train_mode)
            state = self.__agent.pre_process(state)

            self.__env.render(mode=mode)

            if self.__config.get_env_is_atari_flag():
                lives = -1
                new_life = False

            if num_steps is None:
                while True:
                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    if self.__config.get_env_is_atari_flag():
                        if self.__config.get_agent_start_game_action_required():
                            if new_life:
                                action = self.__config.get_agent_start_game_action()

                    self.__env.render(mode=mode)

                    state, reward, done, info = self.__env.step(action)

                    state = self.__agent.pre_process(state)

                    if self.__config.get_env_is_atari_flag():
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

            scores.append([i, reward_total, step])

            recorder.save()

        import numpy as np

        np_scr = np.array(scores)

        mean = np_scr[:, 1].mean()

        return scores, mean

    def play_classic_banana(self, num_episodes=3, score_max=True, score_med=False, trained=True, mode='rgb_array',
                            model_filename=None, num_steps=None):

        recorder = Recorder(header=['episode', 'step', 'action', 'reward', 'reward_total'],
                            session_id=self.__session_id,
                            experiments_path=self.__config.get_app_experiments_path(train_mode=False),
                            model=None)

        if trained or (model_filename is not None):
            self.__agent.current_model.load_state_dict(
                torch.load(model_filename, map_location=lambda storage, loc: storage))

        for i in range(num_episodes):
            scores = []
            reward_total = 0
            step = 0

            brain_name = self.__env.brain_names[0]
            brain = self.__env.brains[brain_name]

            # reset the environment
            env_info = self.__env.reset(train_mode=False)[brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state

            # state = self.__env.reset()
            # self.__env.render(mode=mode)

            # if self.__config.get_current_env_is_atari_flag():
            #     lives = -1
            #     new_life = False

            if num_steps is None:
                while True:
                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    env_info = self.__env.step(action)[brain_name]  # send the action to the environment

                    next_state = env_info.vector_observations[0]  # get the next state
                    reward = env_info.rewards[0]  # get the reward
                    done = env_info.local_done[0]  # see if episode has finished
                    reward_total += reward
                    step += 1
                    state = next_state  # roll over the state to next time step

                    if done:  # exit loop if episode finished
                        print('done')
                        break

                    recorder.record([i, step, action, reward, reward_total])

                    if done:
                        break
            else:
                for j in range(num_steps):

                    if trained:
                        action = self.__agent.act(state)
                    else:
                        action = self.__agent.act(state, eps=1.)

                    env_info = self.__env.step(action)[brain_name]  # send the action to the environment

                    next_state = env_info.vector_observations[0]  # get the next state
                    reward = env_info.rewards[0]  # get the reward
                    done = env_info.local_done[0]  # see if episode has finished
                    reward_total += reward
                    step += 1
                    state = next_state  # roll over the state to next time step

                    if done:  # exit loop if episode finished
                        print('done')
                        break

                    recorder.record([i, step, action, reward, reward_total])

            scores.append([[i, reward_total, step]])

            recorder.save()

        return scores

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
