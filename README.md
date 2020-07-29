placeholder for project

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
