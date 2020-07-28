import pytest
import matplotlib.pyplot as plt
from drl.experiment.analyser import Analyzer
from drl.experiment.config import Config
from drl.experiment.experiment import Experiment
from drl.experiment.explorer import Explorer


class TestAnalyser:

    @pytest.mark.depends(on=['test_play'])
    def test_listPlayExperiments_experimentsExist_returnsExperiments(self):

        config = Config(test=True)
        explorer = Explorer(config=config)
        experiment = Experiment(config)
        analyzer = Analyzer(config=config, session_id=experiment.get_session_id())

        experiments = explorer.list_play_experiments()
        assert len(experiments) > 0

        for experiment in experiments:
            analysis = analyzer.play_analysis(experiment)

            analyzer.log_analysis(analysis)

            assert(len(analysis.keys()) > 0)

    def test_listTrain_Experiments_selectExperiment_performsAnalysis(self):
        config = Config(test=True)
        explorer = Explorer(config=config)
        experiment = Experiment(config)
        analyzer = Analyzer(config=config, session_id=experiment.get_session_id())

        experiments = explorer.list_train_experiments()

        file = analyzer.compare_train_epoch_cols(experiments, 'avg_score')

        assert file is not None

    def test_listTrain_Experiments_selectExperiment_prints_config(self):
        config = Config(test=True)
        explorer = Explorer(config=config)
        experiment = Experiment(config)
        analyzer = Analyzer(config=config, session_id=experiment.get_session_id())

        experiments = explorer.list_train_experiments()

        analyzer.compare_train_config(experiments)






