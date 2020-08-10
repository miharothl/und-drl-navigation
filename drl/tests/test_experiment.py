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

            if config.get_env_type() != 'unity':
                experiment.play_dummy(mode='rgb-array', model=None, num_episodes=3, num_steps=10)

    @pytest.mark.depends(name='test_train')
    def test_train_configExist_canTrain1Episode(self):
        config = Config(test=True)
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_env_type() != 'unity':
                max_steps = 128
                max_episode_steps = 2
                scores = experiment.train(max_steps=max_steps, eval_frequency=16, eval_steps=4,
                                          max_episode_steps=max_episode_steps)

                assert len(scores) == max_steps / max_episode_steps

    @pytest.mark.depends(name='test_train')
    def test_train_unityCconfigExist_canTrain1Episode(self):
        config = Config(test=True)
        experiment = Experiment(config)

        envs = experiment.list_envs()

        for env in envs:
            experiment.set_env(env)

            if config.get_env_type() == 'unity':
                max_steps = 128
                max_episode_steps = 2
                scores = experiment.train(max_steps=max_steps, eval_frequency=16, eval_steps=4,
                                          max_episode_steps=max_episode_steps)

                assert len(scores) == max_steps / max_episode_steps

                break
