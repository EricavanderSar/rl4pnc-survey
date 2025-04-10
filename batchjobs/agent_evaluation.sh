#!/bin/bash
#set job requirements
#SBATCH --job-name="single_agent_eval"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=08:00:00
#SBATCH --output=Evaluation_Results_Case14/Summarize_Eval_Agents_%j.out

AGENT_TYPE="heur" # options "heur" or "rl"
RESDIR=$HOME/HeuristicBaselines/
LIBDIR=$HOME/Rl4Pnc/
CHRONICS="test"
CASE=14
MAXSTEPS=8064

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

j=${SLURM_JOB_ID}
echo "Run agent_evaluation.py..."
time srun python -u scripts/agent_evaluation.py -a $AGENT_TYPE -c $CHRONICS -p $RESDIR -l $LIBDIR -at 0.95 -j $j -lr
#-o -lr -ld -rt 0.8 -s
echo "Done"

echo "Run summarize_evaluation_data.py..."
time srun python -u scripts/summarize_evaluation_data.py -p $RESDIR -c $CASE -m $MAXSTEPS
echo "Done"

