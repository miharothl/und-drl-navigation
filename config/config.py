def get_env_config():
    config = {
        'lunarlander': {
                'id': 'LunarLander-v2',
                'agent': {
                        'action_size': 4,
                        'state_size': 8,
                        'discrete': True,
                        'state_rgb': False
                    },
                'train': {
                        'terminate_score': 300
                    },
            },
        'cartpole': {
                'id': 'CartPole-v1',
                'agent': {
                        'action_size': 2,
                        'state_size': 4,
                        'discrete': True,
                        'state_rgb': False
                    },
                'train': {
                        'terminate_score': 300
                    }
            },
        'banana': {
                'id': 'banana',
                'agent': {
                        'action_size': 4,
                        'state_size': 37,
                        'discrete': True,
                        'state_rgb': False
                    },
                'train': {
                        'terminate_score': 300
                    }
            },
        'spaceinvaders': {
                'id': 'SpaceInvaders-ram-v0',
                'agent': {
                        'action_size': 6,
                        'state_size': 128,
                        'discrete': True,
                        'state_rgb': False,
                    },
                'train': {
                        'terminate_score': 300
                    }
            },
        'spaceinvaders-rgb': {
                'id': 'SpaceInvaders-v0',
                'agent': {
                        'action_size': 6,
                        'state_size': 84,
                        'discrete': True,
                        'state_rgb': True,
                    },
                'train': {
                        'terminate_score': 600
                    }
            },
    }

    return config


def get_app_config():
    config = {
        'path_models': 'models'
    }

    return config
