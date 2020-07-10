def get_env_config():
    config = {
        'lunarlander':
            {
                'id': 'LunarLander-v2',
                'agent':
                    {
                        'action_size': 4,
                        'state_size': 8,
                        'discrete': True
                    },
                'train':
                    {
                    },
            },
        'cartpole':
            {
                'id': 'CartPole-v1',
                'agent':
                    {
                        'action_size': 2,
                        'state_size': 4,
                        'discrete': True
                    },
                'train':
                    {
                    }
            },
        'banana':
            {
                'id': 'banana',
                'agent':
                    {
                        'action_size': 4,
                        'state_size': 37,
                        'discrete': True
                    },
                'train':
                    {
                    }
            },
         'spaceinvaders':
            {
                'id': 'SpaceInvaders-ram-v0',
                'agent':
                    {
                        'action_size': 6,
                        'state_size': 128,
                        'discrete': True
                    },
                'train':
                    {
                    }
            },
    }

    return config


def get_app_config():
    config = {
        'path_models': 'models'
    }

    return config
