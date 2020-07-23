import pytest

from drl.experiment.config import Config
from drl.experiment.experiment import Experiment


class TestExperiment:

    def test_listEnvs_configExist_returnsEnvs(self):

        config = Config(test=True)
        experiment = Experiment(config)

        envs = experiment.list_envs()

        assert len(envs) > 1

    @pytest.mark.depends(name='test_play')
    def test_playDummy_configExist_playsWithDummyAgent(self):

        config = Config(test=True)
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            a = experiment.create_agent()
            e = experiment.create_env()

            assert a is not None
            assert e is not None

            experiment.play_dummy(mode='rgb-array', model=None, num_episodes=3, num_steps=10)

    @pytest.mark.depends(name='test_train')
    def test_train_configExist_canTrain1Episode(self):
        config = Config(test=True)
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            a = experiment.create_agent()
            e = experiment.create_env()

            assert a is not None
            assert e is not None

            num_episodes = 1
            scores = experiment.train(num_episodes=num_episodes)

            assert len(scores) == num_episodes
