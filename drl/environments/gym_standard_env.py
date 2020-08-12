import gym
from gym.spaces import Discrete

from drl import logging
from drl.environments.environment import Environment


class GymStandardEnv(Environment):

    def __init__(self, name, termination_reward):
        self.__env = gym.make(name)
        self.__env.seed(0)
        self.__termination_reward = termination_reward

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)

        if done:
            reward += self.__termination_reward

        new_life = False

        return next_state, reward, done, new_life

    def reset(self, train_mode=False):

        new_life = True
        return self.__env.reset(), new_life

    def render(self, mode):
        self.__env.render(mode=mode)
        pass

    def get_action_space(self):

        isDiscrete = isinstance(self.__env.action_space, Discrete)

        if isDiscrete:
            num_action_space = self.__env.action_space.n
            logging.debug("Env action space is discrete")
            logging.debug("Env action space: {}".format(num_action_space))

        logging.debug("Env observation space: {}".format(self.__env.observation_space))
        pass

    def close(self):
        self.__env.close()
