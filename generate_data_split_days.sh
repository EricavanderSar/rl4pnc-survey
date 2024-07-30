#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=12:00:00
#SBATCH --output=SplitDataPerNDays_%j.out


ENVNAME=l2rpn_case14_sandbox_train #l2rpn_icaps_2021_large_train
NUMBER_DAYS=2

echo "Activate envirnonment"
source activate mahrl_grid2op
export PYTHONPATH=$PYTHONPATH:$PWD

echo "Run code:"
time srun python -u scripts/generate_per_day_scenarios.py -e $ENVNAME -p $HOME/data_grid2op/ -d $NUMBER_DAYS
echo "Done"

