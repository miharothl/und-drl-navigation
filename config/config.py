def get_config():
    config_template = {
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
    }

    return config_template
