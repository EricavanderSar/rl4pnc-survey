#!/bin/bash
#set job requirements
#SBATCH --job-name="eval_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=16:00:00
#SBATCH --output=Evaluate_Agents%j.out


ENVNAME=l2rpn_case14_sandbox_test
RESDIR=$HOME/ray_results/Case14_Sandbox_ActSpaces
LIBDIR=$HOME/mahrl_grid2op/

echo "Activate envirnonment"
source activate mahrl_grid2op
export PYTHONPATH=$PYTHONPATH:$PWD

echo "Run code:"
time srun python -u scripts/multiple_agent_analysis.py -e $ENVNAME -p $RESDIR -l $LIBDIR -w 16
echo "Done"



