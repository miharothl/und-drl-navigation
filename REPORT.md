[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://raw.githubusercontent.com/miharothl/DRLND-Navigation/master/images/lunar-lander.gif  "Lunar Lander"
[image3]: https://raw.githubusercontent.com/miharothl/DRLND-Navigation/master/images/udacity-dueling-q-network.png "Dueling"

# Goal

The goal of the project is to train an agent to navigate and collect bananas in a large world.

![Trained Agent][image1]

Reward +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The task is episodic, and to solve the environment, the agent must get an average score of +13 over 100
consecutive episodes.

# Approach

I started with the base code DQN code provided by [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn)
 [1], which can train an agent to land a lunar lander.

![Lunar Lander][image2]

I adjusted the code to the Unity Banana environment, and the code agent was able to achieve 12.5 scores over 100 consecutive episodes.
I tried to change many different hyperparameters and different deep learning network architectures, and they all resulted in
the decrease in agent performance.

Udacity suggested vanilla DQN [5] algorithm improvements.
 
I started with the implementation of Prioritized Experience Replay [6]. I found some excellent resources [2], [4] and after adding the algorithm found out that
is still couldn't solve the environment.
 
I continued with the next suggested improvement by implementing Double Q Network [7]. That finally solved the environment, and the agent was able to achieve the average scores of +16. I continued with the implementation of
Dueling Q Network [9], which made similar results as Double Q Network algorithm.

The steps that I followed to solve this environment

1. Evaluate the state and action space of the environment
2. Establish a baseline using a random action policy
3. Implement the learning algorithm
4. Run experiments and select the best agent

## Evaluate State & Action Space of the Environment

The state-space has 37 dimensions and contains the agent's velocity, along with the ray-based perception of objects around
the agent's forward direction. Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

* move forward
* move backward
* turn left
* turn right

## Establish Baseline Using Random Action Policy

Before starting the deep reinforcement learning process its good to understand the environment by

1. playing the game manually 
1. playing the game with the agent where actions are selected randomly
 
Random agent achieved scores averaging 0 over 100 consecutive episodes.
 
## Implement Learning Algorithm

The
[agent](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/agents/classic/dqn_agent.py)
and 
[environment](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/environments/unity_env.py)
are created according to the provided
[configuration](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/experiment/config.py)
.
[Recorder](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/experiment/recorder.py)
records the experiment and store the results for later
[analysis](https://github.com/miharothl/DRLND-Navigation/blob/master/analysis.ipynb)
.

The agent interacts with the environment in the
[training loop](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/experiment/trainer.py)
In the exploration phase (higher *Epsilon*) of the training
agent's actions are mostly random. Actions, environment states, dones, and rewards tuples, are stored in the experience
replay buffer. The *Buffer Size* parameter determines the size of the buffer.

Two networks current and target with identical architecture are used to stabilise the DQN learning process. During the learning process, weights of the target network are fixed (or updated more slowly based on parameter *Tau*). DQN and Double Q Learning
uses the same 
[neural network architecture](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/models/classic/model.py)
 while architecture for dueling network is different to accommodate separation state and advantage values.  
 
![Dueling Network][image3]

The loss function is defined as the mean square error of temporal difference error, the difference between the expected
and estimated q values. Adam optimizer minimizes the loss function performing the gradient descent and backpropagation algorithm
using the specified *Learning Rate*.

Learning is performed on every *Update Every* steps, when *Batch Size* of actions, states, dones and rewards tuples are
sampled from the
[replay buffer](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/agents/replay_buffer.py)
either randomly or in case of prioritized experience replay, based on their importance,
determined by the temporal difference error. Prioritized experience replay requires
[segment trees](https://github.com/miharothl/DRLND-Navigation/blob/master/drl/agents/segment_tree.py)
.

During the exploitation phase of the training (lower *Epsilon*) the agent actions are rarely random (*epsilon end* = 0.01)
and mostly based on the estimated Q values calculated by the current neural network.


## Run Experiments and Select Best Agent

[Training](https://github.com/miharothl/DRLND-Navigation/blob/master/navigation.ipynb)
is done using the epochs, consisting of training episodes where epsilon greedy agent is used,
and validation episodes using only trained agent. I used the following training hyper parameters:

|Hyper Parameter           |Value                 |
|:---                      |:---                  |
|Max Steps                 |600000 (2000 episodes)|
|Max Episode Steps         |300                   |
|Evaluation Frequency      |10200  (34 episodes)  |
|Evaluation Steps          |2100   (7 episodes)   |
|Epsilon Decay             |0.995                 |
|Batch Size                |64                    |
|Update Every              |4                     |
|Learning Rate             |0.0001                |
|Tau                       |0.001                 |
|Gamma                     |0.99                  |
|Hidden Layers Units       |[64, 64]              |
|Use Dueling               |False                 |
|Use Double                |True                  |
|Buffer Size               |100000                |
|Use Prioritized Replay    | True                 |
|Prioritized Replay Alpha  | 0.6                  |
|Prioritized Replay Beta0  | 0.4                  |
|Prioritized Replay Eps    | 1e-06                |

 
### Observations

* Both Dueling and Double network perform better than vanilla DQN. There was no significant difference between the Dueling and the Double  Q network.
* Less exploration (eps 0.98 vs. 0.95) didn't result in faster convergence of the optimization.
* Using more neurons (128 vs. 64) in fully connected layers didn't increase the speed of convergence. After 2000 episodes, simpler networks achieved better scores (+1).
* Using more frames (4 vs. 1) didn't improve the speed of convergence or and result of the agent.
* Prioritized experience replay didn't improve the results.

The project is solved in epoch 32 after playing 1088 episodes. The trained agent achieves an average score of 16.51 over 100 episodes.

# Future Work

Deep reinforcement learning is a fascinating and exciting topic. I'll continue to improve my reinforcement learning
laboratory by implementing the Rainbow [9] algorithm and try to reproduce the experiments by Deep Mind on some of the 
legendary Atari games.

# References
  - [1] [Udacity](https://github.com/udacity/deep-reinforcement-learning)
  - [2] [Open AI Baselines](https://github.com/openai/baselines)
  - [3] [RL Advanture](https://github.com/higgsfield/RL-Adventure)
  - [4] [Understanding Prioritized Experience Replay](https://danieltakeshi.github.io/2019/07/14/per/)
  - [5] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
  - [6] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - [7] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
  - [8] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
  - [9] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
