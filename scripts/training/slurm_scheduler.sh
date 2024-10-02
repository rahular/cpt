#!/bin/bash
#SBATCH --account=project_462000353
#SBATCH --job-name=cpt-sv-llama31-base
#SBATCH --nodes=32
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --output=logs/cpt-sv-llama31-base_%j.out
#SBATCH --error=logs/cpt-sv-llama31-base_%j.err
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --time=1-00:00:00

echo "START TIME: $(date)"
sepset -eo pipefail
set -x

mkdir -p logs/separate-logs

# Load your Python environment
# module purge
# module use /appl/local/csc/modulefiles/
# module load pytorch

module purge
module load LUMI/23.09
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240404
export CC=gcc-10
export CXX=g++-10
export PYTHONUSERBASE=/scratch/project_462000353/aralikatte/cpt/.local
export PATH=/scratch/project_462000353/aralikatte/cpt/.local/bin:$PATH

# training hparams
max_steps=20000 # 25000
lr=5e-5
optim="adamw_torch"
beta1=0.9
beta2=0.95

max_seq_len=2048
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
galore_trainable="attn,mlp" # only for galore optimizers; for eg: "galore_adamw"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

deepspeed_config_file=ds_zero2_no_offload.json
pretrained_model=meta-llama/Meta-Llama-3.1-8B # /scratch/project_462000353/aralikatte/cpt/scripts/training/models/Mistral-7B-v0.2
tokenizer_name_or_path=${pretrained_model} # /scratch/project_462000353/aralikatte/cpt/scripts/custom_tokenizers/mistral-0.2-sv-tokenizer-hf

file_list=("/scratch/project_462000353/aralikatte/data/smollm-corpus/fineweb-edu-dedup/[0-40].parquet" # 50
"/scratch/project_462000444/europa/FINAL-DATA/sv/culturax/sv_part_000[0-1][0-9].jsonl"
# "/scratch/project_462000444/europa/FINAL-DATA/sv/hplt/sv-part-[0-9].jsonl"
"/scratch/project_462000444/europa/FINAL-DATA/sv/wikipedia/*.jsonl"
"/scratch/project_462000444/europa/FINAL-DATA/sv/europarl/*.jsonl")

data_cache=/scratch/project_462000353/aralikatte/cpt/scripts/training/vanilla_llama31_data_cache # /scratch/project_462000353/aralikatte/cpt/scripts/training/data_cache_extended
per_device_train_batch_size=2
per_device_eval_batch_size=8
gradient_accumulation_steps=2
output_dir=models/cpt-sv-llama31-base
run_name=cpt-sv-llama31-base

# program to run
PROGRAM="run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --optim ${optim} \
    --adam_beta1 ${beta1} \
    --adam_beta2 ${beta2} \
    --file_list ${file_list[@]} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --do_eval \
    --max_eval_samples 10000 \
    --low_cpu_mem_usage \
    --seed $RANDOM \
    --max_steps ${max_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --block_size ${max_seq_len} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --max_grad_norm 0.5 \
    --report_to wandb \
    --run_name ${run_name} \
    --dataloader_num_workers 0 \
    --use_dora=False \
    --use_rslora=False \
    --use_flash_attention_2 \
    --bf16
    "
    # --num_train_epochs 1 \
    # --overwrite_output_dir \

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
