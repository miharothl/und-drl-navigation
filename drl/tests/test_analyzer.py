import pytest

from drl.experiment.analyser import Analyzer
from drl.experiment.config import Config
from drl.experiment.explorer import Explorer


class TestAnalyser:

    @pytest.mark.depends(on=['test_play'])
    def test_listPlayExperiments_experimentsExist_returnsExperiments(self):

        config = Config(test=True)
        explorer = Explorer(config = config)
        analyzer = Analyzer(config = config)

        experiments = explorer.list_play_experiments()
        assert len(experiments) > 0

        for experiment in experiments:
            analysis = analyzer.play_analysis(experiment)

            analyzer.log_analysis(analysis)

            assert(len(analysis.keys()) > 0)
