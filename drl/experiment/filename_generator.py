import os

from drl.experiment.config import Config
from drl.experiment.experiment import Experiment


class FilenameGenerator:
    def __init__(self, config: Config):
        self.__config = config
        pass

    def get_model_filename(self, episode, score, eps):
        import re
        model_id = re.sub('[^0-9a-zA-Z]+', '', self.__model_id)
        model_id = model_id.lower()
        filename = "{}_{}_{}_{:.2f}_{:.2f}.pth".format(model_id, self.__timestamp, episode, score, eps)

        return os.path.join(self.__config.get_app_experiments_path(), filename)


    def create_session_directory(self, experiment: Experiment):

        experiment.get_timestamp()
        pass
