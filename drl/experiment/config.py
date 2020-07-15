
class Config:
    def __init__(self, current_env='lunarlander'):
        self.__app = self.__set_app_config()
        self.__env = self.__set_env_config()
        self.__current_env = current_env

    def get_app_config(self):
        return self.__app

    def get_current_env(self, current_env):
        return self.__current_env

    def get_envs(self):
        return self.__env.keys()

    def set_current_env(self, current_env):
        self.__current_env = current_env

    def get_current_model_id(self):
        return self.__env[self.__current_env]['id']

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

    def get_current_agent_number_frames_flag(self):
         return self.__env[self.__current_env]['agent']['num_frames']

    def __set_app_config(self):
        config = {
            'path_models': 'models'
        }

        return config

    def __set_env_config(self):
        config = {
            'lunarlander': {
                    'id': 'LunarLander-v2',
                    'agent': {
                            'action_size': 4,
                            'state_size': 8,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
                        },
                    'train': {
                            'terminate_score': 300,
                        },
                },
            'lunarlander-4f': {
                    'id': 'LunarLander-v2',
                    'agent': {
                            'action_size': 4,
                            'state_size': 8,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 4,
                        },
                    'train': {
                            'terminate_score': 300,
                        },
                },
            'cartpole': {
                    'id': 'CartPole-v1',
                    'agent': {
                            'action_size': 2,
                            'state_size': 4,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
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
            'spaceinvaders': {
                    'id': 'SpaceInvaders-ram-v0',
                    'agent': {
                            'action_size': 6,
                            'state_size': 128,
                            'discrete': True,
                            'state_rgb': False,
                            'num_frames': 1,
                        },
                    'train': {
                            'terminate_score': 300,
                        }
                },
            'spaceinvaders-rgb': {
                    'id': 'SpaceInvaders-v0',
                    'agent': {
                            'action_size': 6,
                            'state_size': 84,
                            'discrete': True,
                            'state_rgb': True,
                            'num_frames': 1,
                        },
                    'train': {
                            'terminate_score': 600,
                        }
                },
        }

        return config


