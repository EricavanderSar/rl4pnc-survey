#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --time=16:00:00
#SBATCH --output=Train_Results_Case14/Case14_Baseline_ppo_%j.out
#SBATCH --array=1-5
#SBATCH --ear=off

ENVNAME=l2rpn_case14_sandbox #rte_case14_realistic #l2rpn_icaps_2021_large #
WORKDIR=$TMPDIR/evds_output_dir
RESDIR=Case14_SurveyPaperAT

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, sync with wandb..."
#    srun mkdir -p "$HOME/rl4pnc/runs" && cp -r $WORKDIR/runs $HOME/rl4pnc/
    cd $HOME/ray_results/$RESDIR
    for d in $(ls -t -d */); do cd $d; wandb sync --sync-all; cd ..; done
    exit 1
}

# register the signal handler
trap handle_interrupt TERM

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

#Create output directory on scratch
echo "Copy necessary files"
mkdir $WORKDIR
srun cp -r $HOME/Rl4Pnc/configs $WORKDIR/configs
srun cp -r $HOME/Rl4Pnc/data $WORKDIR/data
#mkdir $WORKDIR/data_grid2op/
#srun find $HOME/data_grid2op -type d -name "${ENVNAME}*" -print0 | xargs -0 -I {} cp -r {} $WORKDIR/data_grid2op/


i=${SLURM_ARRAY_TASK_ID}
j=${SLURM_JOB_ID}
echo "Run code: Task id $i"
  time srun python -u scripts/train_ppo_baseline.py -f configs/$ENVNAME/ppo_baseline_batchjob.yaml -wd $WORKDIR -s $i -j $j
echo "Done"

# Synchronize results with WandB
echo "sync with wandb..."
  cd $HOME/ray_results/$RESDIR
  today=$(date +%Y-%m-%d)
  for d in $(ls -t -d */); do
      if [[ $d == *$today* ]]; then
          cd "$d"
          wandb sync --sync-all
          cd ..
      fi
  done
echo "Update WandB"
  cd $HOME/Rl4Pnc
  time srun python -u scripts/update_wandb.py -p $RESDIR
echo "Finished update"
