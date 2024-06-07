#!/bin/bash

#SBATCH --account=project_462000319
#SBATCH --job-name=cpt-sv-all-multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G \
#SBATCH --gres=gpu:mi250:8
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/cpt-sv-all-multinode-1_%j.out
#SBATCH --error=logs/cpt-sv-all-multinode-1_%j.err
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00

echo "START TIME: $(date)"
set -eo pipefail
set -x

LOG_PATH="logs/main_log.txt"

# Load your Python environment
module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000319/aralikatte/cpt/.local
export PATH=/scratch/project_462000319/aralikatte/cpt/.local/bin:$PATH

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=8503
export WORLD_SIZE=$SLURM_NPROCS

lr=2e-4
optim="adamw_torch"
beta1=0.9
beta2=0.95

max_seq_len=2048
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

deepspeed_config_file=ds_zero2_no_offload.json
pretrained_model=mistralai/Mistral-7B-Instruct-v0.2
tokenizer_name_or_path=${pretrained_model}
dataset_dir=/scratch/project_462000444/europa/FINAL-DATA/sv/
data_cache=./data_cache
per_device_train_batch_size=12
per_device_eval_batch_size=12
gradient_accumulation_steps=5
output_dir=models/cpt-sv-all-mistral-7b

LAUNCHER="torchrun \
    --nnodes $NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank \$SLURM_NODEID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "

PROGRAM="run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --optim ${optim} \
    --adam_beta1 ${beta1} \
    --adam_beta2 ${beta2} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --do_eval \
    --low_cpu_mem_usage \
    --seed $RANDOM \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 50 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --block_size ${max_seq_len} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --max_grad_norm 0.5 \
    --report_to wandb \
    --include_tokens_per_second
    "
    # --use_flash_attention_2 \

export CMD="${LAUNCHER} ${PROGRAM}"
echo $CMD

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
