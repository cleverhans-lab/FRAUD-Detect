#!/bin/bash
#SBATCH --job-name=all-dataset
#SBATCH --time=10:00:00
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G



cd ..


Rscript compute_unfairness_range.R --dataset=$3 --rseed=0 --niters=500 --bbox=$1 --kl=$2 --pos=$SLURM_ARRAY_TASK_ID
             
