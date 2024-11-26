#!/bin/bash
#set job requirements
#SBATCH --job-name="single_agent_eval"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --output=Eval_Agent_Gr_n-1_%j.out

AGENT_TYPE="heur" # options "heur" or "rl"
RESDIR=$HOME/HeuristicBaselines/
LIBDIR=$HOME/Rl4Pnc/
CHRONICS="test"

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

j=${SLURM_JOB_ID}
echo "Run code:"
time srun python -u scripts/agent_evaluation.py -a $AGENT_TYPE -c $CHRONICS -p $RESDIR -l $LIBDIR -at 0.95 -j $j
 #-o -lr -ld -rt 0.8 -s
echo "Done"
