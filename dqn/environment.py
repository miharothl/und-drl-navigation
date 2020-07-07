import gym
import logging

from gym.spaces import Discrete

from dqn.dqn_agent import Agent


class Environment:
    def __init__(self, config):
        self.__config = config
        self.__env = None

    def set_env(self, env):
        self.__env = env

    def get_agent(self):
        action_size = self.__config[self.__env]['agent']['action_size']
        state_size = self.__config[self.__env]['agent']['state_size']

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))

        agent = Agent(state_size=state_size, action_size=action_size, seed=0)
        return agent

    def get_env(self):

        environment =  self.__config[self.__env]['id']

        logging.debug('Environment: {}'.format(environment))

        env = gym.make(environment)
        env.seed(0)

        isDiscrete = isinstance(env.action_space, Discrete)

        if isDiscrete:
            num_action_space = env.action_space.n
            logging.debug("Env action space is discrete")
            logging.debug("Env action space: {}".format(num_action_space))

        logging.debug("Env observation space: {}".format(env.observation_space))

        return env
