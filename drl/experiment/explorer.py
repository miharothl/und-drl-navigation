import glob
import os

from drl.experiment.config import Config


class Explorer:

    def __init__(self, config: Config):
        self.__config = config
        pass

    def list_play_experiments(self):
        return self.__list_experiments(train_mode=False)

    def list_train_experiments(self):
        return self.__list_experiments(train_mode=True)

    def __list_experiments(self, train_mode):

        path = self.__config.get_app_experiments_path(train_mode=train_mode)

        experiments = glob.glob(os.path.join(path, '*'))

        for e in experiments:
            print(e)

        return experiments
