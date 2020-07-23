
import gym
import logging

from gym.spaces import Discrete
from datetime import datetime

# from unityagents import UnityEnvironment

from drl.agents.classic.dqn_agent import DqnAgent
from drl.agents.rgb.dqn_agent_rgb import DqnAgentRgb
from drl.experiment.config import Config
from drl.experiment.player import Player
from drl.experiment.trainer import Trainer


# _Experiments
#   _alias_session
      # _models
      # _logs
      # time, score, eps


class Experiment:
    def __init__(self, config: Config):
        self.__config = config
        self.__timestamp = datetime.now().strftime("%Y%m%dT%H%M")

    def play(self, mode, model, num_episodes=3, trained=True, num_steps=None):
        player = Player(model_id=self.__config.get_current_model_id(),
                        env=self.create_env(),
                        agent=self.create_agent(),
                        config = self.__config,
                        session_id = self.get_session_id())

        player.play(trained=trained,
                    mode=mode,
                    is_rgb=self.__config.get_current_agent_state_rgb_flag(),
                    model_filename=model,
                    num_episodes=num_episodes,
                    num_steps=num_steps)

    def play_dummy(self, mode, model, num_episodes=3, num_steps=None):
        scores = self.play(trained=False,
                  mode=mode,
                  model=model,
                  num_episodes=num_episodes,
                  num_steps=num_steps)

    def train(self, model=None, num_episodes=10000):
        trainer = Trainer(
            config=self.__config,
            session_id=self.get_session_id(),
            model_id=self.__config.get_current_model_id())

        return trainer.train(self.create_agent(),
                             self.create_env(),
                             self.__config.get_current_agent_state_rgb_flag(),
                             model_filename=model,
                             num_episodes=num_episodes)

    def set_env(self, env):
        self.__config.set_current_env(env)

    def create_agent(self):
        action_size = self.__config.get_current_agent_action_size()
        state_size = self.__config.get_current_agent_state_size()
        state_rgb = self.__config.get_current_agent_state_rgb_flag()
        num_frames = self.__config.get_current_agent_number_frames_flag()

        logging.debug("Agent action size: {}".format(action_size))
        logging.debug("Agent state size: {}".format(state_size))
        logging.debug("Agent state RGB: {}".format(state_rgb))

        if state_rgb:
            agent = DqnAgentRgb(state_size=state_size, action_size=action_size, seed=0, num_frames=num_frames)
        else:
            agent = DqnAgent(state_size=state_size, action_size=action_size, seed=0, num_frames=num_frames)

        return agent

    def create_env(self):
        environment = self.__config.get_current_model_id()

        logging.debug('Environment: {}'.format(environment))

        if environment == 'banana':
            # env = UnityEnvironment(file_name="Banana.app")
            # env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")
            # return env
            return None
        else:
            env = gym.make(environment)
            env.seed(0)

            isDiscrete = isinstance(env.action_space, Discrete)

            if isDiscrete:
                num_action_space = env.action_space.n
                logging.debug("Env action space is discrete")
                logging.debug("Env action space: {}".format(num_action_space))

            logging.debug("Env observation space: {}".format(env.observation_space))

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
