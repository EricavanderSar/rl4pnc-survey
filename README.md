# Reinforcement Learning for Power Network Control

This repository contains the code for the paper "Optimizing Power Grid Topologies with Reinforcement Learning: A Survey 
of Methods and Challenges" by E. van der Sar et al. 

If you use this repository in your research, please cite the following paper:

```bibtex TOOD
```


## Setup
### Installation
```sh
# Clone the repository
git clone https://github.com/EricavanderSar/rl4pnc-survey
cd rl4pnc
# Optionally create conda environment (or use pipenv)
conda env create -n rl4pnc python=3.10  
conda activate rl4pnc
# Install the required packages
pip install -e .
```

### lightsim2grid installation
The package uses lightsim2grid for the power grid simulation. Which can be installed as follows: \
```sh
pip install lightsim2grid  
```
However, in case you are experiencing the following problem:\
https://github.com/BDonnot/lightsim2grid/issues/55 \
Follow the steps from https://lightsim2grid.readthedocs.io/en/latest/install_from_source.html#install-python 
also described below:
```sh
git clone https://github.com/BDonnot/lightsim2grid.git
cd lightsim2grid
git checkout v0.9.2
git submodule init
git submodule update
make
pip install -U pybind11
pip install -U .
```

## Getting Started
This is still TO DO!

### 0. Setup Weights and Biases
Optionally

### 1. Split the data

### 2. Configuration of training

### 3. Train an agent
To train an agent run script/train_ppo_baseline.py for example:

```sh
WORKDIR=/path/to/workdir
ENVNAME="l2rpn_case14_sandbox"
i=0 # Choose seed number
j="jobid" # Choose jobid
time srun python -u scripts/train_ppo_baseline.py -f configs/$ENVNAME/ppo_baseline_batchjob.yaml -wd $WORKDIR -s $i -j $j
```

### 3.1. Synchronize with Weights and Biases

```sh
# Synchronize results with WandB
echo "sync with wandb..."
  cd $HOME/ray_results/$RESDIR #or wherever you have your results
  for d in $(ls -t -d */); do
      cd "$d"
      wandb sync --sync-all
      cd ..
  done
# Run extra scripts such that objects like the reward function can be filtered on
echo "Update WandB"
  cd $HOME/Rl4Pnc
  time srun python -u scripts/update_wandb.py -p $RESDIR
echo "Finished update"
```

### 4. Evaluate agent(s)
