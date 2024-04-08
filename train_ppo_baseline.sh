#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --output=ParamTunCase14_ppo_baseline_%j.out


ENVNAME=rte_case14_realistic
WORKDIR=$TMPDIR/evds_output_dir

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, copying output directory from scratch to home..."
    srun mkdir -p "$HOME/mahrl/runs" && cp -r $WORKDIR/runs $HOME/mahrl/
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
echo "copy output to home dir"
srun mkdir -p "$HOME/mahrl/runs" && cp -r $WORKDIR/runs $HOME/mahrl/

