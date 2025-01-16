#!/bin/bash
#set job requirements
#SBATCH --job-name="eval_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=16:00:00
#SBATCH --output=Results_RL_Evaluations/Evaluate_Agents_%j.out


RESDIR=$HOME/ray_results/Case14_SurveyPaperObs
LIBDIR=$HOME/Rl4Pnc/
CHRONICS="test"
CASE=14
MAXSTEPS=8064

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

j=${SLURM_JOB_ID}
echo "Run multiple_agent_analysis.py..."
time srun python -u scripts/multiple_agent_analysis.py -p $RESDIR -l $LIBDIR -w 16 -c $CHRONICS -j $j -at 0.95 -lr
# -o (opponent) -ld (line_dics) -lr (line_reconnect) -rt (reset_topo) -s (simulate)
echo "Done"

echo "Run summarize_evaluation_data.py..."
time srun python -u scripts/summarize_evaluation_data.py -p $RESDIR -c $CASE -m $MAXSTEPS
echo "Done"




