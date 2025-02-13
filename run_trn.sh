#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --job-name=czhenguo
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
# srun  --kill-on-bad-exit=1  run_inference.sh
srun  --kill-on-bad-exit=1  run_mainline.sh
# srun  --kill-on-bad-exit=1 --nodes=1 --jobid=181 --exclusive run_mainline.sh
