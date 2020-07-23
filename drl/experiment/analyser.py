import os
import pandas as pd
from typing import Dict


from drl.experiment.config import Config


class Analyzer:

    def __init__(self, config: Config):
        self.__config = config
        pass

    def play_analysis(self, path_to_experiment) -> Dict:

        # path = self.__config.get_app_experiments_path(train_mode=False)
        path = os.path.join(path_to_experiment, 'log.csv')

        df = pd.read_csv(path)

        analysis = {}
        analysis['number_of_episodes'] = df['episode'].max() + 1
        analysis['average_number_of_steps'] = df.groupby('episode').count()['step'].mean()
        analysis['average_reward_per_step'] = df.groupby('episode').mean()['reward'].mean()
        analysis['average_reward'] = df.groupby('episode').max()['reward_total'].mean()

        """

        :return:
            number_of_episodes
            average_number_of_steps
            average_reward
            average_number_of_actions
        """
        return analysis

    def log_analysis(self, analysis: Dict):
        for key in analysis.keys():
            print("{}: {}".format(key, analysis[key]))

    def train_analysis(self):
        """

        :return:
        """
        return {}




