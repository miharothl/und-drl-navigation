[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Deep Reinforcement Learning - Udacity - Project 1: Navigation

The goal of the project is to train an agent to navigate and collect bananas in a large world.

![Trained Agent][image1]

Reward +1 is provided for collecting a yellow banana, and reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around
agent's forward direction. Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

* move forward.
* move backward.
* turn left.
* turn right.

The task is episodic, and in order to solve the environment, agent must get an average score of +13 over 100
consecutive episodes.

## Environment

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `env/python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/miharothl/lab-drlnd-navigation.git
cd lab-drlnd-navigation/env/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

6. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
# How to Use Reinforcement Learning Lab

To get get help 
```
./rlab -h
```
List of supported environments
```
./rlab -l
```

# Development

List of make targets
```
make
```

To run tests
```
make test
```

To clean tests output
```
make clean 
```







```
conda create --name drlnd python=3.6
source activate drlnd

cd env/python

pip install .

python -m ipykernel install --user --name drlnd --display-name "drlnd"

install gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e .

dev env
pip install pytest-depends
pip install opencv-python

rlab -h
```
