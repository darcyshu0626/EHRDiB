#!/bin/bash
#SBATCH --job-name=duke_ehrdib
#SBATCH --array=0-1
#SBATCH --account=pro00110219
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/pro00110219/ss1688/EHRDiB/logs/%x_%j.out
#SBATCH --error=/data/pro00110219/ss1688/EHRDiB/logs/%x_%j.err

set -e

task=completion  # Options: synthesis, completion.
representations=(cat num)

cd /data/pro00110219/ss1688/EHRDiB
module load miniforge
export EHRDIB_ENV=/home/ss1688/software/conda/envs/ehrdib

python -B -u main.py run --config "${1:-configs/duke/${task}.yaml}" \
    --representation "${representations[${SLURM_ARRAY_TASK_ID:?}]}" --device cuda:0
