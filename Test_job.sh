#!/bin/bash
#set job requirements
#SBATCH --job-name="bugtest_ray"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --output=Output_bug_test_%j.out

echo "Activate envirnonment"
source activate rl4pnc
export PYTHONPATH=$PYTHONPATH:$PWD

echo "Run code..."
time srun python -u Debug_Ray.py

echo "Done"