import json
import os
from pathlib import Path

import pandas as pd
from typing import Dict
import matplotlib
import matplotlib.pyplot as plt


from drl.experiment.config import Config


class Analyzer:

    def __init__(self, config: Config, session_id):
        self.__config = config
        self.__session_id = session_id
        pass

    def play_analysis(self, path_to_experiment) -> Dict:

        # path = self.__config.get_app_experiments_path(train_mode=False)
        path = os.path.join(path_to_experiment, 'log.csv')

        df = pd.read_csv(path)

        analysis = {}
        analysis['number_of_episodes'] = df['episode'].max() + 1
        analysis['average_number_of_steps'] = df.groupby('episode').count()['step'].mean()
        analysis['average_reward_per_step'] = df.groupby('episode').mean()['reward'].mean()
        analysis['average_reward'] = df.groupby('episode').max()['reward_total'].sum()

        """

        :return:
            number_of_episodes
            average_number_of_steps
            average_reward
            average_number_of_actions
        """
        return analysis

    def compare_train_config(self, path_to_experiments):

        for path_to_experiment in path_to_experiments:
            path = os.path.join(path_to_experiment, 'config.json')

            with open(path, 'r') as fp:
                cfg = json.load(fp)

            print(path_to_experiment)
            print(json.dumps(cfg, indent=4, sort_keys=True))

        pass


    def compare_train_epoch_cols(self, path_to_experiments, compare_col, plot=False):
        max_x = 0

        for path_to_experiment in path_to_experiments:
            path = os.path.join(path_to_experiment, 'epoch-log.csv')

            df = pd.read_csv(path, index_col=0)
            if df.shape[0] > max_x:
                max_x = df.shape[0]

        df_main = pd.DataFrame(list(range(max_x)))

        for path_to_experiment in path_to_experiments:
            path = os.path.join(path_to_experiment, 'epoch-log.csv')

            df_tmp = pd.read_csv(path, index_col=0)

            df_tmp = df_tmp[[compare_col]]

            df_tmp.columns = [(path_to_experiment.rsplit('/', 1)[1])]

            df_main = df_main.join(df_tmp)


        df_main = df_main[df_main.columns.difference([0])]

        ax = df_main.plot()

        ax.set_xlabel('epoch')
        ax.set_ylabel(compare_col)

        from matplotlib import pyplot
        fig = ax.get_figure()

        path = os.path.join(self.__config.get_app_analysis_path(), self.__session_id)
        Path(path).mkdir(parents=True, exist_ok=True)

        path = os.path.join(path, compare_col +'.png' )

        if plot:
            plt.show()
            return None
        else:
            matplotlib.use('Agg')
            fig.savefig(path)
            return path

    def compare_train_epoch_score(self, path_to_experiment, plot=False):

        path = os.path.join(path_to_experiment, 'epoch-log.csv')

        df_tmp = pd.read_csv(path, index_col=0)

        df_tmp = df_tmp[['avg_score', 'avg_val_score']]

        ax = df_tmp.plot()

        ax.set_xlabel('epoch')
        ax.set_ylabel('score')

        from matplotlib import pyplot
        fig = ax.get_figure()

        path = os.path.join(self.__config.get_app_analysis_path(), self.__session_id)
        Path(path).mkdir(parents=True, exist_ok=True)

        path = os.path.join(path, 'score.png' )
        
        if plot:
            plt.show()
            return None
        else:
            matplotlib.use('Agg')
            fig.savefig(path)
            return path

    def log_analysis(self, analysis: Dict):
        for key in analysis.keys():
            print("{}: {}".format(key, analysis[key]))





