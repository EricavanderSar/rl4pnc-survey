#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=12:00:00
#SBATCH --output=SplitData_TrainValTest_%j.out


ENVNAME=l2rpn_icaps_2021_large #rte_case14_realistic #

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

echo "Run code:"
time srun python -u scripts/generate_train_val_test_split.py -e $ENVNAME -p $HOME/data_grid2op/
echo "Done"

