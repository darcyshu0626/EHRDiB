#!/bin/bash
#SBATCH --job-name=mimic_ehrdib
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/hpc/group/honglab/ss1688/EHRDiB/logs/%x_%j.out
#SBATCH --error=/hpc/group/honglab/ss1688/EHRDiB/logs/%x_%j.err

set -e

task=completion  # Options: synthesis, completion.

cd /hpc/group/honglab/ss1688/EHRDiB
source /hpc/group/honglab/ss1688/miniconda3/bin/activate base
export EHRDIB_ENV=/hpc/group/honglab/ss1688/miniconda3/envs/ehrdib

python -B -u main.py run --config "${1:-configs/mimic/${task}.yaml}" --device cuda:0
