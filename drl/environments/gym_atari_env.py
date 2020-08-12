import gym
from gym.spaces import Discrete

from drl import logging
from drl.environments.environment import Environment


class GymAtariEnv(Environment):

    def __init__(self, name, termination_reward=0, lost_life_reward=-1):
        self.__env = gym.make(name)
        self.__env.seed(0)
        self.__termination_reward = termination_reward
        self.__lost_life_reward = lost_life_reward

        self.__lives = -1
        self.__new_life = False

    def step(self, action):
        next_state, reward, done, info = self.__env.step(action)

        if info['ale.lives'] > self.__lives:
            self.__lives = info['ale.lives']
            self.__new_life = True
        elif info['ale.lives'] < self.__lives:
            self.__lives = info['ale.lives']
            self.__new_life = True
            reward += self.__lost_life_reward
        else:
            self.__new_life = False

        if done:
            reward += self.__termination_reward

        return next_state, reward, done, self.__new_life

    def reset(self, train_mode=False):
        self.__lives = -1
        self.__new_life = True

        return self.__env.reset(), self.__new_life

    def render(self, mode):
        self.__env.render(mode=mode)

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
