import os
import json
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path

class Recorder:
    def __init__(self, header: List, experiments_path,  session_id, model, log_prefix = '', configuration={}):
        self.__header = header
        self.__parameters = []
        self.__session_id = session_id
        self.__model = model
        self.__experiments_path = experiments_path
        self.__log_prefix = log_prefix
        self.__configuration = configuration
        pass

    def get_header(self) -> List:
        return self.__header

    def record(self, parameters: List):
        self.__parameters.append(parameters)

    def save(self):
        self.__save_log()
        self.__save_config()

    def load(self) -> Tuple[Dict[str, str], pd.DataFrame]:
        log = self.__load_log()
        config = self.__load_config()

        return (config, log)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.__parameters, columns=self.__header)

    def __save_config(self):
        session_path = os.path.join(self.__experiments_path, self.__session_id)
        Path(session_path).mkdir(parents=True, exist_ok=True)
        action_path = os.path.join(session_path, 'config.json')

        data = {'session_id': self.__session_id, 'model': self.__model}
        data['configuration'] = self.__configuration

        with open(action_path, 'w') as fp:
            json.dump(data, fp)

    def __save_log(self):
        session_path = os.path.join(self.__experiments_path, self.__session_id)
        Path(session_path).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(session_path, self.__log_prefix + 'log.csv')

        df = self.get_dataframe()
        df.to_csv(log_path, index=False)

    def __load_config(self) -> Dict[str,str]:
        file_path = os.path.join(self.__experiments_path, self.__session_id, 'config.json')

        with open(file_path, 'r') as fp:
            return json.load(fp)

    def __load_log(self) -> pd.DataFrame:
        file_path = os.path.join(self.__experiments_path, self.__session_id, self.__log_prefix + 'log.csv')
        return pd.read_csv(file_path)
