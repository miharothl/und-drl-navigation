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

    def get_current_model_id(self):
        return self.__env[self.__current_env]['id']

    def get_current_env_is_atari_flag(self):
        return self.__env[self.__current_env]['env']['is_atari']

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
                            'is_atari': False,
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
                            'terminate_score': 300,
                        },
                },
            'lunarlander-4f': {
                    'id': 'LunarLander-v2',
                    'env': {
                            'is_atari': False,
                        },
                    'agent': {
                            'action_size': 4,
                            'state_size': 8,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 4,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'terminate_score': 300,
                        },
                },
            'cartpole': {
                    'id': 'CartPole-v1',
                    'env': {
                        'is_atari': False,
                    },
                    'agent': {
                            'action_size': 2,
                            'state_size': 4,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'terminate_score': 300,
                        }
                },
            # 'banana': {
            #         'id': 'banana',
            #         'agent': {
            #                 'action_size': 4,
            #                 'state_size': 37,
            #                 'discrete': True,
            #                 'state_rgb': False,
            #                 'num_frames': 1,
            #             },
            #         'train': {
            #                 'terminate_score': 300,
            #             }
            #     },
            'breakout': {
                   'id': 'Breakout-ram-v4',
                   'env': {
                        'is_atari': True,
                   },
                   'agent': {
                           'action_size': 3,
                           'state_size': 128,
                           'discrete': True,
                           'state_rgb': False,
                           'num_frames': 4,
                           'state_offset': 1,
                           'start_game_action_required': True,
                           'start_game_action': 1,
                   },
                   'train': {
                       'terminate_score': 1000,
                   }
               },
            'spaceinvaders': {
                    'id': 'SpaceInvaders-ram-v0',
                    'env': {
                        'is_atari': True,
                    },
                    'agent': {
                            'action_size': 6,
                            'state_size': 128,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 4,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'terminate_score': 1000,
                        }
                },
            'spaceinvaders-rgb': {
                    'id': 'SpaceInvaders-v0',
                    'env': {
                        'is_atari': True,
                    },
                    'agent': {
                            'action_size': 6,
                            'state_size': 84,
                            'discrete': True,
                            'state_rgb': True,
                            'num_frames': 4,
                            'state_offset': 0,
                            'start_game_action_required': False,
                            'start_game_action': 0,
                        },
                    'train': {
                            'terminate_score': 1000,
                        }
                },
        }

        return config


