from drl.experiment.config import Config
from drl.experiment.experiment import Experiment
from drl.experiment.recorder import Recorder


class TestRecorder:

    def test_record_recordsParameters_multipleSavesOverwrites(self):

        config = Config(test=True)
        session_id = Experiment(config).get_session_id()
        experiments_path = config.get_app_experiments_path(train_mode=False)
        model='model123'

        header = ['episode', 'step', 'action', 'reward']

        recorder = Recorder(
            header=header,
            experiments_path=experiments_path,
            session_id=session_id,
            model=model
        )

        header_result = recorder.get_header()

        assert all([a == b for a, b in zip(header, header_result)])

        parameters1 = [1,1,0,0]
        parameters2 = [1,2,1,0]
        parameters3 = [1,3,0,10]

         # episode 1
        recorder.record(parameters1)
        recorder.record(parameters2)
        recorder.record(parameters3)

        recorder.save()

        df = recorder.get_dataframe()
        (config, log) = recorder.load()

        assert df.shape[0] == log.shape[0]

        # episode 2
        recorder.record(parameters1)
        recorder.record(parameters2)
        recorder.record(parameters3)

        recorder.save()

        df = recorder.get_dataframe()
        (config, log) = recorder.load()

        assert df.shape[0] == log.shape[0]
        assert config['session_id'] == session_id
        assert config['model'] == model







