# Multi-Agent Hierarchical Reinforcement Learning for Power Network Topology Control

This repository contains the code for experiments conducted for my master thesis realized at University of Groningen and TenneT.

### Dependencies 

The `pyproject.toml` file can be used to install the dependencies using `pip install -e ".[dev]`.

### Scripts

The scripts can be used to generate all needed scenarios and action spaces, as well as to train and evaluate agents.

##### Generate initial resources

Using `python generate_train_val_test_split.py`, the train-validation-test split can be generated.

Flags:
- `-e`: Describe the environment name (e.g. rte_case5_example).
- `-p`: Specify the path in which the environment is downloaded. Default is /home/data_grid2op/.
- `-t`: Specify the percentage of testing data. Default is 20%.
- `-v`: Specify the percentage of testing data. Default is 10%.

Using `python generate_per_day_scenarios.py`, the full length scenarios can be split up into days.

Flags:
- `-c`: Specify the path to the config file that specifies the environment set-up.
- `-s`: Specify the path for the new scenarios to be saved.
- `-d`: Specify the number of days that should be contained in a single scenario. Note: Selecting a number that does not exactly splits up the day leads to the last day having another length.

Using `python develop_action_spaces.py`, action spaces can be generated based on the asymmetrical (asymmetry), N-0 secure (medha) or N-1 (tennet) secure principles.

Flags:
- `-e`: Describe the environment name (e.g. rte_case5_example).
- `-a`: Describe the action space that should be generated (asymmetry/medha/tennet). Default: all.
- `-s`: Specify the path for the new action spaces to be saved.

##### Training
Using `python train_ppo_custom.py`, single agents using PPO can be trained.

Flags:
- `-c`: Specify the path to the config file that specifies the environment, training and agent set-up.
- `-r`: Specify the path to a saved checkpoint in case you want to retrain an agent.

Using `python train_ppo_hrl.py`, all hierarchical multi-agent architectures can be trained.

Flags:
- `-c`: Specify the path to the config file that specifies the environment, training and agent set-up.
- `-m`: Specify the type of coordinator you want to use (rl, rlv, capa, random, sample, argmax). Default: rl.
- `-l`: Specify the type of regional agents you want to use (rl, rlv, greedy). Default: rl.

##### Evaluation
Using `python evaluate_custom_model.py`, all baselines, benchmarks and architectures can be evaluated.

Flags:
- `-c`: Specify the path to the config file that specifies the environment and agent set-up.
- `-n` (bool): Add if you wish to evaluate the performance of the do-nothing agent.
- `-g` (bool): Add if you wish to evaluate the performance of the greedy agent.
- `-capa` (bool): Add if you wish to evaluate the performance of the hierarchical capa-greedy agent.
- `-random` (bool): Add if you wish to evaluate the performance of the hierarchical random-greedy agent.
- `-f`: Add if you wish to evaluate a trained architecture, specify the path in which the trained instance is saved.
- `-cp`: Add if you wish to evaluate a trained architecture, specify the name of the checkpoint you wish to use. Default: checkpoint_000000.
- `-ma` (bool): Add if you wish to evaluate a trained architecture, in case the architecture has a hierarchical multi-agent setup.

Using `python faster_analysis.py`, the evaluated scenarios are analysed and resulting data is saved as csv files in the same folder.

Flags:
- `-c`: Specify the path to the config file that specifies the environment and agent set-up.
- `-f`: Specify the path in which the scenario output files are saved.
