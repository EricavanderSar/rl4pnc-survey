#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=42:00:00
#SBATCH --output=PostFCN_Case14_ppo_baseline_%j.out


ENVNAME=rte_case14_realistic
WORKDIR=$TMPDIR/evds_output_dir
RESDIR= Case14_PostFCN

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, sync with wandb..."
#    srun mkdir -p "$HOME/mahrl/runs" && cp -r $WORKDIR/runs $HOME/mahrl/
    cd $HOME/ray_results/$RESDIR
    for d in $(ls -t -d */); do cd $d; wandb sync --sync-all; cd ..; done
    exit 1
}

# register the signal handler
trap handle_interrupt TERM

echo "Activate envirnonment"
source activate mahrl_grid2op
export PYTHONPATH=$PYTHONPATH:$PWD

#Create output directory on scratch
echo "Copy necessary files"
mkdir $WORKDIR
srun cp -r $HOME/mahrl_grid2op/configs $WORKDIR/configs
srun cp -r $HOME/mahrl_grid2op/data $WORKDIR/data
mkdir $WORKDIR/data_grid2op/
srun find $HOME/data_grid2op -type d -name "${ENVNAME}*" -print0 | xargs -0 -I {} cp -r {} $WORKDIR/data_grid2op/


echo "Run code:"
time srun python -u scripts/train_ppo_baseline.py -f configs/$ENVNAME/ppo_baseline_batchjob.yaml -wd $WORKDIR
echo "Done"

#Copy output directory from scratch to home
echo "sync with wandb..."
cd $HOME/ray_results/$RESDIR
for d in $(ls -t -d */);
do
  cd $d; wandb sync --sync-all; cd ..;
done

