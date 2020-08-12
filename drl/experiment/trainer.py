import os
from pathlib import Path

import torch
import numpy as np
from collections import deque
import sys
import logging

from drl.experiment.config import Config
from drl.experiment.recorder import Recorder


class Trainer:
    def __init__(self, model_id, config: Config, session_id, path_models='models'):
        self.__model_id = model_id
        self.__config = config
        self.__session_id = session_id

    def get_model_filename(self, episode, score, val_score, eps):

        session_path = os.path.join(self.__config.get_app_experiments_path(train_mode=True), self.__session_id)
        Path(session_path).mkdir(parents=True, exist_ok=True)

        import re
        model_id = re.sub('[^0-9a-zA-Z]+', '', self.__config.get_current_env())
        model_id = model_id.lower()
        filename = "{}_{}_{}_{:.2f}_{:.2f}_{:.2f}.pth".format(model_id, self.__session_id, episode, score, val_score, eps)

        model_path = os.path.join(session_path, filename)

        return model_path

    def train(self, agent, env, is_rgb,
              model_filename=None,
              max_steps=10000,
              max_episode_steps=18000,
              eval_frequency=10000,
              eval_steps=10000,
              eps_decay=0.99,
              is_human_flag=False):
        if is_rgb:
            return self.dqn_rgb(agent, env, model_filename, n_episodes=max_steps, eval_steps=eval_steps)
        else:
            return self.dqn_classic(agent, env, model_filename,
                                    max_steps=max_steps,
                                    max_episode_steps=max_episode_steps,
                                    eval_frequency=eval_frequency,
                                    eval_steps=eval_steps,
                                    eps_decay=eps_decay,
                                    is_human_flag=is_human_flag
                                    )

    def select_model_filename(self, model_filename=None):
        if model_filename is not None:
            path = os.path.join(self.__path_models, model_filename)
            return path

    def dqn_classic(self, agent, env, model_filename=None,
                    max_steps=300000000,
                    max_episode_steps=18000,
                    eval_frequency=2000,
                    eval_steps=10000,
                    is_human_flag=False,
                    eps_start=1.0,
                    eps_end=0.01,
                    eps_decay=0.9990, terminate_soore=800.0):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon

        loss_window = deque(maxlen=100)
        pos_reward_ratio_window = deque(maxlen=100)
        neg_reward_ratio_window = deque(maxlen=100)

        # start with pre-trained model
        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.current_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.target_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            eps = 0.78

        epoch_recorder = Recorder(
            header=['epoch', 'avg_score', 'avg_val_score', 'epsilon', 'avg_loss', 'beta'],
            session_id=self.__session_id,
            experiments_path=self.__config.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='epoch-',
            configuration = self.__config.get_current_env_config()
        )

        episode_recorder = Recorder(
            header=['step', 'episode', 'epoch', 'epoch step', 'epoch_episode', 'episode step', 'score', 'epsilon', 'beta',
                    'avg_pos_reward_ratio', 'avg_neg_reward_ratio', 'avg_loss'],
            session_id=self.__session_id,
            experiments_path=self.__config.get_app_experiments_path(train_mode=True),
            model=None,
            log_prefix='episode-',
            configuration=self.__config.get_current_env_config()
        )

        EVAL_FREQUENCY = eval_frequency
        EVAL_STEPS = eval_steps
        MAX_STEPS = max_steps
        MAX_EPISODE_STEPS = max_episode_steps

        step = 0
        epoch = 0
        episode = 0

        while step < MAX_STEPS:

            epoch_step = 0

            ################################################################################
            # Training
            ################################################################################

            terminal = True
            epoch_episode = 0

            while (epoch_step < EVAL_FREQUENCY) and (step < MAX_STEPS):

                for episode_step in range(MAX_EPISODE_STEPS):

                    if epoch_step >= EVAL_FREQUENCY:
                        break
                    elif step >= MAX_STEPS:
                        break

                    if terminal:
                        terminal = False

                        state, new_life = env.reset(train_mode=not is_human_flag)
                        state = agent.pre_process(state)

                        score = 0
                        epoch_episode += 1

                    action = agent.act(state, eps)

                    if self.__config.get_agent_start_game_action_required():
                        if new_life:
                            action = self.__config.get_agent_start_game_action()

                    action = action + self.__config.get_agent_state_offset()

                    if is_human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    action = action - self.__config.get_agent_state_offset()

                    if done:
                        reward += self.__config.get_env_terminate_reward()


                    pos_reward_ratio, neg_reward_ratio, loss, beta = agent.step(state, action, reward, next_state, done)

                    if loss is not None:
                        loss_window.append(loss)
                        pos_reward_ratio_window.append(pos_reward_ratio)
                        neg_reward_ratio_window.append(neg_reward_ratio)

                    step += 1
                    epoch_step += 1

                    state = next_state
                    score += reward

                    if done:
                        break

                    logging.debug(
                        'Step: {}\tEpisode: {}\tEpoch: {}\tEpoch Step: {}\tEpoch Episode: {}\tEpisode Step: {}\tScore: {:.2f}'
                        '\tEpsilon: {:.2f}\tAvg Pos Reward Ratio: {:.3f}\tAvg Neg Reward Ratio: {:.3f}\tLoss {:.6f}'
                            .format(step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps,
                                    np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                                    np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                                    np.mean(loss_window) if len(loss_window) > 0 else 0))
                logging.warning(
                    'Step: {}\tEpisode: {}\tEpoch: {}\tEpoch Step: {}\tEpoch Episode: {}\tEpisode Step: {}\tScore: {:.2f}'
                    '\tEpsilon: {:.2f}\tAvg Pos Reward Ratio: {:.3f}\tAvg Neg Reward Ratio: {:.3f}\tLoss {:.6f}'
                        .format(step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps,
                                np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                                np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                                np.mean(loss_window) if len(loss_window) > 0 else 0))

                episode_recorder.record([step, episode, epoch, epoch_step, epoch_episode, episode_step, score, eps, beta,
                                         np.mean(pos_reward_ratio_window) if len(pos_reward_ratio_window) > 0 else 0,
                                         np.mean(neg_reward_ratio_window) if len(neg_reward_ratio_window) > 0 else 0,
                                         np.mean(loss_window) if len(loss_window) > 0 else 0])

                episode += 1

                if step <= MAX_STEPS:
                    scores_window.append(score)  # save most recent score

                eps = max(eps_end, eps_decay * eps)  # decrease epsilon

                sys.stdout.flush()

                episode_recorder.save()

                terminal = True

            ################################################################################
            # Validation
            ################################################################################

            val_step = 0

            val_scores_window = deque(maxlen=100)  # last 100 scores

            terminal = True
            epoch_val_episode = 0

            while val_step < EVAL_STEPS:

                for episode_val_step in range(MAX_EPISODE_STEPS):

                    if val_step >= EVAL_STEPS:
                        break

                    if terminal:

                        terminal = False

                        state, new_life = env.reset(train_mode=not is_human_flag)

                        state = agent.pre_process(state)
                        score = 0
                        epoch_val_episode += 1

                    action = agent.act(state, eps)

                    if self.__config.get_agent_start_game_action_required():
                        if new_life:
                            action = self.__config.get_agent_start_game_action()

                    action = action + self.__config.get_agent_state_offset()

                    if is_human_flag:
                        env.render(mode='human')

                    next_state, reward, done, new_life = env.step(action)

                    next_state = agent.pre_process(next_state)

                    if done:
                        reward += self.__config.get_env_terminate_reward()

                    val_step += 1

                    state = next_state
                    score += reward

                    if done:
                        break

                    logging.debug(
                        'Epoch: {}\tVal Step: {}\tEpoch Val Episode: {}\tEpisode Step: {}\tVal Score: {:.2f}\tEpsilon: {:.2f}'
                            .format(epoch, val_step, epoch_val_episode, episode_val_step, score, eps))

                logging.warning(
                    'Epoch: {}\tVal Step: {}\tEpoch Val Episode: {}\tEpisode Step: {}\tVal Score: {:.2f}\tEpsilon: {:.2f}'
                        .format(epoch, val_step, epoch_val_episode, episode_val_step, score, eps))

                if val_step < EVAL_STEPS:
                    val_scores_window.append(score)  # save most recent score

                sys.stdout.flush()

                terminal = True

            logging.info(
                'Epoch {}\t Score: {:.2f}\t Val Score: {:.2f}\tEpsilon: {:.2f}'.format(epoch, np.mean(scores_window),
                                                                                         np.mean(val_scores_window),
                                                                                         eps))

            epoch_recorder.record([epoch, np.mean(scores_window), np.mean(val_scores_window), eps, np.mean(loss_window), beta])
            epoch_recorder.save()

            model_filename = self.get_model_filename(epoch, np.mean(scores_window), np.mean(val_scores_window), eps )

            torch.save(agent.current_model.state_dict(), model_filename)

            epoch += 1

        env.close()

        return scores_window

    def dqn_rgb(self, agent, env, model_filename=None, n_episodes=10000, eval_steps=1000, eps_start=1.0, eps_end=0.05,
                eps_decay=0.995, terminate_soore=800.0):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon

        if (model_filename is not None):
            filename = self.select_model_filename(model_filename=model_filename)
            agent.current_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            agent.target_model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
            eps = 0.78

        for i_episode in range(1, n_episodes + 1):
            state = env.reset()

            from drl.image import imshow
            state = imshow(state)

            score = 0
            for t in range(eval_steps):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)

                from drl.image import imshow
                next_state = imshow(next_state)

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    agent.check_memory()
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score

            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            model_filename = self.get_model_filename(i_episode, np.mean(scores_window), eps)

            sys.stdout.flush()

            print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
                  end="")
            if i_episode % 100 == 0:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                    eps))
                torch.save(agent.current_model.state_dict(), model_filename)

            if i_episode % 100 == 0:
                plot_filename = self.get_plot_filename(i_episode, np.mean(scores_window), eps)
                # self.plot(scores, plot_filename)

            if np.mean(scores_window) >= terminate_soore:
                print('\nEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode - 100,
                                                                                    np.mean(scores_window), eps))
                torch.save(agent.current_model.state_dict(), model_filename)
                self.plot(scores)
                break

        return scores
