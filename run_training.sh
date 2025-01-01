#!/bin/bash

# Set resource limits
ulimit -n 65535
ulimit -s unlimited

# Set environment variables
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch/DDP Configuration
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Clean up any existing processes
pkill -9 python || true

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Wait for system to stabilize
sleep 5

# Print system information
echo "System Information:"
echo "==================="
nvidia-smi
echo
echo "Memory Information:"
echo "=================="
free -h
echo

echo "Starting training..."

# Run with torchrun
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train-new-pytorch-light.py 