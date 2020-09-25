[//]: # (Image References)


[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Deep Reinforcement Learning - Project Navigation

The goal of the project is to train an agent to navigate and collect bananas in a large world.

![Trained Agent][image1]

Reward +1 is provided for collecting a yellow banana, and reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around
agent's forward direction. Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

* move forward
* move backward
* turn left
* turn right

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
git clone https://github.com/miharothl/DRLND-Navigation.git
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
  
If not Mac OSX copy environment into `env/unity` and update configuration `drl/experiment/config.py`

```
'banana':
{
    'id': 'env/unity/mac/banana',
    'env': {
...
```
    
## How to Use Reinforcement Learning Lab

To get get help 
```
./rlab -h
```
To get list of supported environments
```
./rlab -l
```
To train banana environment
```
./rlab -e banana -t -vvv
```
To play banana environment with a dummy agent
```
./rlab -vvv -e banana -p
```
To play banana environment with a trained agent
```
./rlab -vvv -e banana -p -f _experiments/train/banana-20200812T0605/banana_banana-20200812T0605_58_16.52_16.33_0.01.pth
```

# Development

To run tests
```
make clean && make test
```

