#!/bin/bash
#SBATCH --account=project_462000353
#SBATCH --job-name=parallel-data-processing
#SBATCH --nodes=32
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --output=logs/process-data_%j.out
#SBATCH --error=logs/process-data_%j.err
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --time=0-03:00:00

echo "START TIME: $(date)"
sepset -eo pipefail
set -x

mkdir -p logs/separate-logs

module purge
module load LUMI/23.09
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240404
export CC=gcc-10
export CXX=g++-10
export PYTHONUSERBASE=/scratch/project_462000353/aralikatte/cpt/.local
export PATH=/scratch/project_462000353/aralikatte/cpt/.local/bin:$PATH

block_size=2048
# dataset_dir=/scratch/project_462000444/europa/FINAL-DATA/el # /scratch/project_462000353/aralikatte/cpt/data/sv_en
file_list=("/scratch/project_462000353/aralikatte/data/smollm-corpus/fineweb-edu-dedup/*.parquet"
"/scratch/project_462000444/europa/FINAL-DATA/sv/*/*.jsonl")
data_cache=/scratch/project_462000353/aralikatte/cpt/scripts/training/vanilla_llama31_data_cache
# /scratch/project_462000353/aralikatte/cpt/scripts/custom_tokenizers/mistral-0.2-sv-tokenizer-hf
# /scratch/project_462000353/aralikatte/cpt/scripts/training/models/Mistral-7B-v0.2
model_name_or_path=meta-llama/Meta-Llama-3.1-8B
preprocessing_num_workers=32

# program to run
PROGRAM="process_data.py \
    --block_size ${block_size} \
    --file_list ${file_list[@]} \
    --data_cache_dir ${data_cache} \
    --model_name_or_path ${model_name_or_path} \
    --preprocessing_num_workers ${preprocessing_num_workers} \
    "

GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8503
export WORLD_SIZE=$SLURM_NTASKS
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

SING_BIND="/scratch/project_462000353,/scratch/project_462000444"

srun \
    --label \
    singularity exec \
    -B "$SING_BIND" \
    -B "$PWD" \
    "$SIF" \
    bash ./slurm_runner.sh \
    $PROGRAM
