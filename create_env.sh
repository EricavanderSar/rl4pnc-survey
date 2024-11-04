#!/bin/bash
#Set job requirements
#SBATCH --job-name="create_env"
#SBATCH -t 00:20:00
#SBATCH --output=init_env_%j.out

echo "*** START create env job ***"

module load 2022
module load Anaconda3/2022.05

echo "Start updating conda"
conda init
conda update conda
echo "Create rl4pnc env"
time conda create -n rl4pnc python=3.10
echo "********** activate Env **********"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD
python -m pip install .
pip install -e .
echo "************** done: environment packages installed *************"

echo "************* install lightsim2grid ***************"
git clone https://github.com/BDonnot/lightsim2grid.git      
cd lightsim2grid
git checkout v0.7.3
git submodule init
git submodule update
make
pip install -U pybind11
pip install -U .

echo "done"

