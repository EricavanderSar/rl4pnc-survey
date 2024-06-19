#!/bin/bash
#set job requirements
#SBATCH --job-name="mahrl_create_act_space"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --output=Create_action_space_%j.out


ENVNAME=l2rpn_icaps_2021_large #rte_case14_realistic #
SAVE_PATH=$HOME/mahrl_grid2op/data/action_spaces/

echo "Activate envirnonment"
source activate mahrl_grid2op
export PYTHONPATH=$PYTHONPATH:$PWD

echo "Run code:"
time srun python -u scripts/develop_action_spaces.py -e $ENVNAME -s $SAVE_PATH -a medha -dn -sh "" -rf 1.0 -w 16 -g -ps 100 -i 200
echo "Done"