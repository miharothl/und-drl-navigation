import torch
from typing import Tuple

from drl.models.classic.model import DqnDueling2Hidden, Dqn2Hidden, Dqn3Hidden, Dqn4Hidden


class ModelFactory:

    def __init__(self):
        pass

    @staticmethod
    def create(type, fc_units, num_frames, state_size, action_size, dueling, seed, device)\
            -> Tuple[torch.nn.Module, torch.nn.Module]:

        supported_types = ['classic', 'rgb']

        if type not in supported_types:
            assert ("Type {} not supported. Supported types: {}".format(type, supported_types))

        if type == "classic":

            if len(fc_units) == 2:

                if dueling:
                    current_model = DqnDueling2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                    target_model = DqnDueling2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                else:
                    current_model = Dqn2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
                    target_model = Dqn2Hidden(
                        state_size * num_frames, action_size, seed,
                        fc1_units=fc_units[0],
                        fc2_units=fc_units[1]
                    ).to(device)
            elif len(fc_units) == 3:
                current_model = Dqn3Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2]).to(device)

                target_model = Dqn3Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2]).to(device)
            elif len(fc_units) == 4:
                current_model = Dqn4Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2],
                    fc4_units=fc_units[3]).to(device)

                target_model = Dqn4Hidden(
                    state_size * num_frames, action_size, seed,
                    fc1_units=fc_units[0],
                    fc2_units=fc_units[1],
                    fc3_units=fc_units[2],
                    fc4_units=fc_units[3]).to(device)

            return current_model, target_model
