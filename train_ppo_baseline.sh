#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=00:20:00
#SBATCH --output=test_train_ppo_baseline_%j.out


# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, copying output directory from scratch to home..."
    cp -r "$TMPDIR"/evds_output_dir/result $HOME/marl/
    exit 1
}

# register the signal handler
trap handle_interrupt TERM

echo "Activate envirnonment"
source activate mahrl_grid2op
export PYTHONPATH=$PYTHONPATH:$PWD

#Create output directory on scratch
echo "Copy necessary files"
mkdir "$TMPDIR"/evds_output_dir
srun cp -r $HOME/mahrl_grid2op/configs $TMPDIR/evds_output_dir/configs
srun cp -r $HOME/mahrl_grid2op/data $TMPDIR/evds_output_dir/data
srun cp -r $HOME/data_grid2op/ $TMPDIR/evds_output_dir/data_grid2op


echo "Run code:"
time srun python -u scripts/train_ppo_baseline.py -f "$TMPDIR"/evds_output_dir/configs/rte_case14_realistic/ppo_baseline_batchjob.yaml
done
echo "Done"

#Copy output directory from scratch to home
echo "copy output to home dir"
srun cp -r "$TMPDIR"/evds_output_dir/run $HOME/mahrl/

