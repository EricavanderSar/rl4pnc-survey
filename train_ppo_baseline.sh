#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=00:20:00
#SBATCH --output=test_train_ppo_baseline_%j.out

WORKDIR=$TMPDIR/evds_output_dir

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, copying output directory from scratch to home..."
    cp -r $WORKDIR/result $HOME/mahrl/
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
srun cp -r $HOME/data_grid2op/ $WORKDIR/data_grid2op


echo "Run code:"
time srun python -u scripts/train_ppo_baseline.py -f $WORKDIR/configs/rte_case14_realistic/ppo_baseline_batchjob.yaml
done
echo "Done"

#Copy output directory from scratch to home
echo "copy output to home dir"
srun cp -r $WORKDIR/run $HOME/mahrl/

