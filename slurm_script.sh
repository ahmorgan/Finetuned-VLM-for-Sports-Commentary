#!/bin/bash
#SBATCH --job-name=4152-vlm-finetune
#SBATCH --time=24:00:00
#SBATCH --output=output_vlmft.log
#SBATCH --error=error_vlmft.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amorga94@charlotte.edu
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --mem=24G

nvidia-smi
source ~/miniconda3/etc/profile.d/conda.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"

conda init
conda activate vlm_finetune
export CUDA_HOME=$CONDA_PREFIX

module load 

cd ~/final_project

unset NVCC_PREPEND_FLAGS
unset NVCC_APPEND_FLAGS
unset CC
unset CXX

#accelerate launch --config_file context_parallel_2gpu.yaml vlm_training.py
python -u vlm_training.py

echo "Job finished at: $(date)"