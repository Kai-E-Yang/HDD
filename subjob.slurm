#!/bin/sh -l

#SBATCH -A xxx
#SBATCH --partition=xxx
#SBATCH --nodes=1 
#SBATCH --time=8:00:00
#SBATCH --job-name TEST
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --error=test-%A.err ## %A - filled with jobid
#SBATCH --output=test-%A.out ## %A - filled with jobid
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=xxx


# Print the hostname of the compute node on which this job is running.
/bin/hostname

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -u run.py --config config.json > out_print.log 2>&1