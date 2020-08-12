import gym
import logging

from gym.spaces import Discrete
from datetime import datetime

from unityagents import UnityEnvironment

from drl.agents.classic.dqn_agent import DqnAgent
from drl.agents.rgb.dqn_agent_rgb import DqnAgentRgb
from drl.environments.gym_atari_env import GymAtariEnv
from drl.environments.gym_standard_env import GymStandardEnv
from drl.environments.unity_env import UnityEnv
from drl.experiment.config import Config
from drl.experiment.player import Player
from drl.experiment.trainer import Trainer


class Experiment:
    def __init__(self, config: Config):
        self.__config = config
        self.__timestamp = datetime.now().strftime("%Y%m%dT%H%M")

    def play(self, mode, model, num_episodes=3, trained=True, num_steps=None):

        with Player(model_id=self.__config.get_current_model_id(),
                    env=self.create_env(),
                    agent=self.create_agent(),
                    config=self.__config,
                    session_id=self.get_session_id()) as player:
            return player.play(trained=trained,
                               mode=mode,
                               is_rgb=self.__config.get_agent_state_rgb_flag(),
                               model_filename=model,
                               num_episodes=num_episodes,
                               num_steps=num_steps)

    def play_dummy(self, mode, model, num_episodes=3, num_steps=None):
        self.play(trained=False,
                  mode=mode,
                  model=model,
                  num_episodes=num_episodes,
                  num_steps=num_steps)

    def train(self, model=None, max_steps=None, eval_frequency=None, eval_steps=None, max_episode_steps=None):

        trainer = Trainer(
            config=self.__config,
            session_id=self.get_session_id(),
            model_id=self.__config.get_current_model_id()
        )

        if max_steps is None:
            max_steps = self.__config.get_train_max_steps()

        if max_episode_steps is None:
            max_episode_steps = self.__config.get_train_max_episodes_steps()

        if eval_steps is None:
            eval_steps = self.__config.get_train_eval_steps()

        if eval_frequency is None:
            eval_frequency = self.__config.get_train_train_eval_frequency()

        eps_decay = self.__config.get_train_epsilon()

        is_human_flag = self.__config.get_train_is_human_flag()

        return trainer.train(self.create_agent(),
                             self.create_env(),
                             self.__config.get_agent_state_rgb_flag(),
                             model_filename=model,
                             max_steps=max_steps,
                             max_episode_steps=max_episode_steps,
                             eval_frequency=eval_frequency,
                             eval_steps=eval_steps,
                             is_human_flag=is_human_flag,
                             eps_decay=eps_decay,
                             )

    def set_env(self, env):
        self.__config.set_current_env(env)

    def create_agent(self):
        action_size = self.__config.get_agent_action_size()
        state_size = self.__config.get_agent_state_size()
        state_rgb = self.__config.get_agent_state_rgb_flag()
        num_frames = self.__config.get_agent_num_frames()

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))
        logging.debug("Agent state RGB: {}".format(state_rgb))

        if state_rgb:
            agent = DqnAgentRgb(state_size=state_size, action_size=action_size, seed=None, num_frames=num_frames)
        else:
            agent = DqnAgent(seed=0, cfg=self.__config)

        return agent

    def create_env(self):
        environment = self.__config.get_current_model_id()

        type = self.__config.get_env_type()

        if type == 'gym_standard':
            env = GymStandardEnv(name=environment,
                                 termination_reward=self.__config.get_env_terminate_reward())
        elif type == 'gym_atari':
            env = GymAtariEnv(name=environment,
                              termination_reward=self.__config.get_env_terminate_reward())
        elif type == 'unity':
            env = UnityEnv(name=environment,
                           termination_reward=self.__config.get_env_terminate_reward())
        else:
            raise Exception("Environment {} type not supported".format(environment))

        return env

    def list_envs(self):
        envs = self.__config.get_envs()
        for e in envs:
            print(e)

        return envs

    def get_timestamp(self):
        return self.__timestamp

    def get_session_id(self):
        return "{}-{}".format(
            self.__config.get_current_env(),
            self.get_timestamp()
        )
