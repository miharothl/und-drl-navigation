import os


class Config:
    def __init__(self, current_env='lunarlander', test=False):
        self.__app = self.__set_app_config()
        self.__env = self.__set_env_config()
        self.__current_env = current_env
        self.__test = test

    def get_app_config(self):
        return self.__app

    def get_current_env(self):
        return self.__current_env

    def get_envs(self):
        return self.__env.keys()

    def set_current_env(self, current_env):
        self.__current_env = current_env

    def get_current_env_config(self):
        return self.__env[self.__current_env]

    def get_current_model_id(self):
        return self.__env[self.__current_env]['id']

    def get_current_env_is_atari_flag(self):
        return self.__env[self.__current_env]['env']['is_atari']

    def get_current_env_type(self):
        return self.__env[self.__current_env]['env']['type']

    def get_current_env_terminate_reward(self):
        return self.__env[self.__current_env]['env']['terminate_reward']

    def get_current_env_train_max_steps(self):
         return self.__env[self.__current_env]['train']['max_steps']

    def get_current_env_train_max_episodes_steps(self):
        return self.__env[self.__current_env]['train']['max_episode_steps']

    def get_current_env_train_train_eval_frequency(self):
        return self.__env[self.__current_env]['train']['eval_frequency']

    def get_current_env_train_eval_steps(self):
        return self.__env[self.__current_env]['train']['eval_steps']

    def get_current_env_train_epsilon(self):
        return self.__env[self.__current_env]['train']['epsilon']

    def get_current_env_train_learning_rate(self):
        return self.__env[self.__current_env]['train']['learning_rate']

    def get_current_env_train_tau(self):
        return self.__env[self.__current_env]['train']['tau']

    def get_current_env_train_gamma(self):
        return self.__env[self.__current_env]['train']['gamma']

    def get_current_env_train_neural_network(self):
        return self.__env[self.__current_env]['train']['neural_network']

    def get_current_env_train_is_human_flag(self):
        return self.__env[self.__current_env]['train']['human_flag']

    def get_current_num_frames(self):
        return self.__env[self.__current_env]['agent']['num_frames']

    def get_current_is_rgb(self):
        return self.__env[self.__current_env]['agent']['state_rgb']

    def get_current_agent_action_size(self):
        return self.__env[self.__current_env]['agent']['action_size']

    def get_current_agent_state_size(self):
        return self.__env[self.__current_env]['agent']['state_size']

    def get_current_agent_state_rgb_flag(self):
        return self.__env[self.__current_env]['agent']['state_rgb']

    def get_current_agent_state_offset(self) -> int:
        return int(self.__env[self.__current_env]['agent']['state_offset'])

    def get_current_agent_start_game_action_required(self) -> bool:
        return self.__env[self.__current_env]['agent']['start_game_action_required']

    def get_current_agent_start_game_action(self) -> int:
        return int(self.__env[self.__current_env]['agent']['start_game_action'])

    def get_current_agent_number_frames_flag(self):
        return self.__env[self.__current_env]['agent']['num_frames']


    def get_app_analysis_path(self, train_mode=True):

        if self.__test:
            if train_mode:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], 'analysis')
            else:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], 'analysis')
        else:
            if train_mode:
                return os.path.join(self.__app['path_experiments'], 'analysis')
            else:
                return os.path.join(self.__app['path_experiments'], 'analysis')

    def get_app_experiments_path(self, train_mode=True):

        if self.__test:
            if train_mode:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], self.__app['path_train'])
            else:
                return os.path.join(self.__app['path_tests'], self.__app['path_experiments'], self.__app['path_play'])
        else:
            if train_mode:
                return os.path.join(self.__app['path_experiments'], self.__app['path_train'])
            else:
                return os.path.join(self.__app['path_experiments'], self.__app['path_play'])

    def __set_app_config(self):
        config = {
            'path_experiments': '_experiments',
            'path_tests': '_tests',
            'path_play': 'play',
            'path_train': 'train',
        }

        return config

    def __set_env_config(self):
        config = {
            'lunarlander': {
                    'id': 'LunarLander-v2',
                    'env': {
                            'type': 'gym_standard',
                            'is_atari': False,
                            'terminate_reward': 0,
                        },
                    'agent': {
                            'action_size': 4,
                            'state_size': 8,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'max_steps': 1000000,
                            'max_episode_steps': 1000,
                            'eval_frequency': 20000,
                            'eval_steps': 3000,
                            'epsilon': 0.995,
                            'human_flag': False,
                            'learning_rate': 0.0001,
                            'tau': 0.001,
                            'gamma': 0.99,
                            'neural_network': [64,64]
                        },
                },
            'cartpole': {
                    'id': 'CartPole-v1',
                    'env': {
                        'type': 'gym_standard',
                        'is_atari': False,
                        'terminate_reward': -50,
                    },
                    'agent': {
                            'action_size': 2,
                            'state_size': 4,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 5,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'max_steps': 1000000,
                            'max_episode_steps': 1000,
                            'eval_frequency': 5000,
                            'eval_steps': 500,
                            'epsilon': 0.99995,
                            'human_flag': False,
                            'learning_rate': 0.0001,
                            'tau': 0.001,
                            'gamma': 0.99,
                            'neural_network': [64,64]
                        }
                },
            'banana':
                {
                    'id': 'env/unity/mac/banana',
                    'env': {
                        'type': 'unity',
                        'is_atari': False,
                        'terminate_reward': 0,
                    },
                    'agent': {
                            'action_size': 4,
                            'state_size': 37,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'max_steps':   600000,
                            'max_episode_steps': 1000,
                            'eval_frequency': 10200,
                            'eval_steps': 2100,
                            'epsilon': 0.995,
                            'human_flag': False,
                            'learning_rate': 0.0001,
                            'tau': 0.001,
                            'gamma': 0.99,
                            'neural_network': [64,64]
                        }
                },
            'breakout': {
                'id': 'Breakout-ram-v4',
                'env': {
                    'type': 'gym_atari',
                    'is_atari': True,
                    'terminate_reward': 0,
                },
                'agent': {
                    'action_size': 3,
                    'state_size': 128,
                    'discrete': True,
                    'state_rgb': False,
                    'num_frames': 4,
                    'state_offset': 1,
                    'start_game_action_required': True,
                    'start_game_action': 0,
                },
                'train': {
                    'max_steps': 1000000,
                    'max_episode_steps': 2000,
                    'eval_frequency': 20000,
                    'eval_steps': 3000,
                    'epsilon': 0.995,
                    'human_flag': False,
                    'learning_rate': 0.0001,
                    'tau': 0.001,
                    'gamma': 0.99,
                    'neural_network': [256, 128, 64, 64]
                }
            },
        }

        return config


