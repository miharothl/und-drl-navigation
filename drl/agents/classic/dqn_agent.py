import numpy as np
import random
import torch
import torch.optim as optim
from collections import deque

from drl.agents.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from drl.agents.schedules import LinearSchedule
from drl.experiment.config import Config
from drl.models.model_factory import ModelFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, seed, cfg: Config):
        """Initialize an Agent object.

        Params
        ======
            seed (int): random seed
            cfg (Config): configration
        """

        # training parameters
        self.batch_size = 64  # minibatch size
        self.update_every = 4  # how often to update the network
        self.learning_rate = cfg.get_train_learning_rate()  # learning rate
        self.tau = cfg.get_train_tau()  # for soft update of target parameters
        self.gamma = cfg.get_current_env_train_gamma()  # discount factor

        # agent parameters
        self.state_size = cfg.get_agent_state_size()
        self.action_size = cfg.get_agent_action_size()

        # replay memory parameters
        self.buffer_size = cfg.get_replay_memory_buffer_size()
        self.prioritized_replay = cfg.get_replay_memory_prioritized_replay_flag()
        self.prioritized_replay_alpha = cfg.get_replay_memory_prioritized_replay_alpha()
        self.prioritized_replay_beta0 = cfg.get_replay_memory_prioritized_replay_beta0()
        self.prioritized_replay_eps = cfg.get_replay_memory_prioritized_replay_eps()
        self.total_timesteps = cfg.get_train_max_steps()
        self.prioritized_replay_beta_iters = None

        # network parameters
        nn_cfg = cfg.get_neural_network_hiden_layers()
        dueling = cfg.get_neural_network_dueling_flag()
        self.double_dqn = cfg.get_neural_network_double_flag()
        self.__num_frames = cfg.get_agent_num_frames()

        # Q-Network
        self.current_model, self.target_model = ModelFactory.create(
            type='classic',
            fc_units=nn_cfg,
            num_frames= self.__num_frames,
            state_size= self.state_size,
            action_size= self.action_size,
            dueling= dueling,
            seed=seed,
            device=device)

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.learning_rate)

        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.total_timesteps
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.prioritized_replay_beta0,
                                                final_p=1.0)
        else:
            self.memory = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

        self.step_i = 0

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.__frames_queue = deque(maxlen=self.__num_frames)

    def pre_process(self, raw_state):
        if len(self.__frames_queue) == 0:
            for i in range(self.__num_frames):
                self.__frames_queue.append(raw_state)

        self.__frames_queue.append(raw_state)

        state = np.concatenate(self.__frames_queue)

        return state

    def check_memory(self):
        total = self.buffer_size
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

        # state = self.pre_process(state)
        # next_state = self.pre_process(next_state)

        self.memory.add(state, action, reward, next_state, done)

        pos_reward_ratio = None
        neg_reward_ratio = None
        loss = None
        beta = None

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:

                if self.prioritized_replay:
                    beta = self.beta_schedule.value(self.step_i)
                    experience = self.memory.sample(self.batch_size, beta=beta)
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)
                else:
                    experiences = self.memory.sample(self.batch_size)
                    obses_t, actions, rewards, obses_tp1, dones = experiences
                    weights, batch_idxes = np.ones_like(rewards), None
                    exp = (obses_t, actions, rewards, obses_tp1, dones, weights)

                pos_reward_ratio, neg_reward_ratio, loss, td_error = self.learn(exp, self.gamma)

                if self.prioritized_replay:
                    new_priorities = np.abs(td_error) + self.prioritized_replay_eps
                    self.memory.update_priorities(batch_idxes, new_priorities)

        self.step_i += 1

        return pos_reward_ratio, neg_reward_ratio, loss, beta

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = self.pre_process(state)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.current_model.eval()
        with torch.no_grad():
            action_values = self.current_model(state)
        self.current_model.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long().unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1)
        weights = torch.from_numpy(weights).float().unsqueeze(1)

        if self.double_dqn:
            q_values = self.current_model(states)
            next_q_values = self.current_model(next_states)
            next_q_state_values = self.target_model(next_states)

            q_value = q_values.gather(1, actions).squeeze(1)

            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        else:  # dqn, dueling
            q_values = self.current_model(states)
            next_q_values = self.target_model(next_states)

            q_value = q_values.gather(1, actions).squeeze(1)
            next_q_value = next_q_values.max(1)[0]

        q_value = q_value.unsqueeze(1)
        next_q_value = next_q_value.unsqueeze(1)

        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value).pow(2) * weights

        td_error = loss
        td_error = td_error.squeeze(1)
        td_error = td_error.detach().numpy()

        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.current_model, self.target_model, self.tau)

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
