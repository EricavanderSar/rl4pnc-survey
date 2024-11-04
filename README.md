# Reinforcement Learning for Power Network Control

## Getting started
### Create conda environment
```sh
conda env create -n rl4pnc python=3.10  
conda activate rl4pnc
pip install -e ".[dev] 
```

### lightsim2grid installation
In case you are experiencing the following problem:\
https://github.com/BDonnot/lightsim2grid/issues/55 \
Follow the steps from https://lightsim2grid.readthedocs.io/en/latest/install_from_source.html#install-python
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