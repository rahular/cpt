#!/bin/bash

# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf /dev/shm/*
    rocm-smi || true
fi
sleep 2

export FI_CXI_DEFAULT_CQ_SIZE=262144
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1

export TORCH_EXTENSIONS_DIR=torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run application
python "$@" \
    > >(tee logs/separate-logs/${SLURMD_NODENAME}-${SLURM_PROCID}.out) \
    2> >(tee logs/separate-logs/${SLURMD_NODENAME}-${SLURM_PROCID}.err)
