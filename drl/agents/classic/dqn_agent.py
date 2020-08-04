import numpy as np
import random
from collections import namedtuple, deque

from drl.agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from drl.agents.replay_buffer_udacity import ReplayBufferUdacity
from drl.agents.schedules import LinearSchedule
from drl.experiment.config import Config
from drl.models.classic.model import QNetwork2, QNetwork1, QNetwork3

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
# GAMMA = 0.99  # discount factor
# TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, config: Config, num_frames=1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.__config = config

        self.__TAU = self.__config.get_current_env_train_tau()
        self.__LR = self.__config.get_current_env_train_learning_rate()
        self.__GAMMA = self.__config.get_current_env_train_gamma()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        nn_cfg = self.__config.get_current_env_train_neural_network()

        if len(nn_cfg) == 2:
            self.qnetwork_local = QNetwork1(state_size * num_frames, action_size, seed,
                                            fc1_units=nn_cfg[0],
                                            fc2_units=nn_cfg[1]
                                            ).to(device)

            self.qnetwork_target = QNetwork1(state_size * num_frames, action_size, seed,
                                             fc1_units=nn_cfg[0],
                                             fc2_units=nn_cfg[1]
                                             ).to(device)
        elif len(nn_cfg) == 3:
            self.qnetwork_local = QNetwork2(state_size * num_frames, action_size, seed,
                                            fc1_units=nn_cfg[0],
                                            fc2_units=nn_cfg[1],
                                            fc3_units=nn_cfg[2]).to(device)

            self.qnetwork_target = QNetwork2(state_size * num_frames, action_size, seed,
                                             fc1_units=nn_cfg[0],
                                             fc2_units=nn_cfg[1],
                                             fc3_units=nn_cfg[2]).to(device)
        elif len(nn_cfg) == 4:
            self.qnetwork_local = QNetwork3(state_size * num_frames, action_size, seed,
                                            fc1_units=nn_cfg[0],
                                            fc2_units=nn_cfg[1],
                                            fc3_units=nn_cfg[2],
                                            fc4_units=nn_cfg[3]).to(device)

            self.qnetwork_target = QNetwork3(state_size * num_frames, action_size, seed,
                                             fc1_units=nn_cfg[0],
                                             fc2_units=nn_cfg[1],
                                             fc3_units=nn_cfg[2],
                                             fc4_units=nn_cfg[3]).to(device)

        # Q-Network
        # self.qnetwork_local = QNetwork1(state_size * num_frames, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork1(state_size * num_frames, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.__LR)

        # Replay memory
        # PRB
        # self.memory = ReplayBufferUdacity(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Create the replay buffer
        self.prioritized_replay = True
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta_iters = None
        total_timesteps = self.__config.get_current_env_train_max_steps()
        prioritized_replay_beta0 = 0.4
        self.prioritized_replay_eps = 1e-6

        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE)
            self.beta_schedule = None

        self.step_i = 0

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.__frames = deque(maxlen=num_frames)

        self.__num_frames = num_frames

    def preprocess(self, raw_state):

        if len(self.__frames) == 0:

            for i in range(self.__num_frames):
                self.__frames.append(raw_state)

        self.__frames.append(raw_state)

        state = np.concatenate(self.__frames)

        return state

    def check_memory(self):
        total = BUFFER_SIZE
        current = len(self.memory)

        action_counter = [0 for i in range(self.action_size)]
        for e in self.memory.memory:
            action_counter[e.action] = action_counter[e.action] + 1

        print("\n\ttotal: {} current: {}".format(total, current))

        actions = "\t"
        for i in range(self.action_size):
            actions = actions + "action{}: {} \t".format(i, action_counter[i])

        print(actions)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state = self.preprocess(state)
        next_state = self.preprocess(next_state)

        self.memory.add(state, action, reward, next_state, done)

        pos_reward_ratio = None
        neg_reward_ratio = None
        loss = None

        beta = None



        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # PRB
                # experiences = self.memory.sample()

                if self.prioritized_replay:

                    beta =  self.beta_schedule.value(self.step_i)
                    experience = self.memory.sample(BATCH_SIZE, beta=beta)
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)
                else:
                    experiences = self.memory.sample(BATCH_SIZE)
                    obses_t, actions, rewards, obses_tp1, dones = experiences
                    weights, batch_idxes = np.ones_like(rewards), None
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)

                pos_reward_ratio, neg_reward_ratio, loss, td_error = self.learn(exp, self.__GAMMA)

                if self.prioritized_replay:
                    new_priorities = np.abs(td_error) + self.prioritized_replay_eps
                    self.memory.update_priorities(batch_idxes, new_priorities)

        self.step_i += 1

        return (pos_reward_ratio, neg_reward_ratio, loss, beta)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self.preprocess(state)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # return random.choice(np.arange(-6, 6))
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        # # PRB
        # states = torch.from_numpy(states).float()
        # actions = torch.from_numpy(actions).long()
        # actions = actions.unsqueeze(1)
        # rewards = torch.from_numpy(rewards).float()
        # rewards = rewards.unsqueeze(1)
        # next_states = torch.from_numpy(next_states).float()
        # dones = torch.from_numpy(dones.astype(np.uint8)).float()
        # dones = dones.unsqueeze(1)
        #
        # # print(weights.shape)
        # weights = torch.from_numpy(weights).float()
        # weights = weights.unsqueeze(1)
        #
        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #
        # # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        #
        # # Compute loss
        # # loss = F.mse_loss(Q_expected, Q_targets)
        #
        # #  PER
        # # loss = (torch.FloatTensor(weights) * F.mse_loss(q_expected, q_targets)).mean()
        # # loss = (torch.FloatTensor(weights) * 100000 * F.smooth_l1_loss(Q_expected, Q_targets)).mean()
        #
        # loss = (Q_expected - Q_targets).pow(2) * weights
        #
        # td_error = loss
        # td_error = td_error.squeeze(1)
        # td_error = td_error.detach().numpy()
        #
        # loss = loss.mean()

        # PRB
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        actions = actions.unsqueeze(1)
        rewards = torch.from_numpy(rewards).float()
        rewards = rewards.unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones.astype(np.uint8)).float()
        dones = dones.unsqueeze(1)

        # print(weights.shape)
        weights = torch.from_numpy(weights).float()
        weights = weights.unsqueeze(1)

        q_values = self.qnetwork_local(states)
        next_q_values = self.qnetwork_local(next_states)
        next_q_state_values = self.qnetwork_target(next_states)

        q_value = q_values.gather(1, actions).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        q_value = q_value.unsqueeze(1)
        next_q_value = next_q_value.unsqueeze(1)

        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value).pow(2) * weights

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)

        #  PER
        # loss = (torch.FloatTensor(weights) * F.mse_loss(q_expected, q_targets)).mean()
        # loss = (torch.FloatTensor(weights) * 100000 * F.smooth_l1_loss(Q_expected, Q_targets)).mean()

        # loss = (Q_expected - Q_targets).pow(2) * weights

        td_error = loss
        td_error = td_error.squeeze(1)
        td_error = td_error.detach().numpy()

        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.__TAU)

        return float(torch.sum(rewards > 0)) / rewards.shape[0], float(torch.sum(rewards < 0)) / rewards.shape[
            0], loss.item(), td_error

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
