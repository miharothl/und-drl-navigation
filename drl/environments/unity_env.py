import gym
from gym.spaces import Discrete
from unityagents import UnityEnvironment

from drl import logging
from drl.environments.environment import Environment


class UnityEnv(Environment):

    def __init__(self, name, termination_reward):
        self.__env = UnityEnvironment(file_name=name)
        self.__brain_name = self.__env.brain_names[0]
        self.__termination_reward = termination_reward

        # env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")

    def step(self, action):
        env_info = self.__env.step(action)[self.__brain_name]  # send the action to the environment

        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished

        if done:
            reward += self.__termination_reward

        new_life = False

        return next_state, reward, done, new_life

    def render(self, mode):
        pass

    def reset(self, train_mode=False):
        brain_name = self.__env.brain_names[0]
        # brain = self.__env.brains[brain_name]

        env_info = self.__env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state

        new_life = True

        return state, new_life

    def get_action_space(self):

        # isDiscrete = isinstance(self.__env.action_space, Discrete)
        #
        # if isDiscrete:
        #     num_action_space = self.__env.action_space.n
        #     logging.debug("Env action space is discrete")
        #     logging.debug("Env action space: {}".format(num_action_space))
        #
        # logging.debug("Env observation space: {}".format(self.__env.observation_space))
        pass

    def close(self):
        self.__env.close()
