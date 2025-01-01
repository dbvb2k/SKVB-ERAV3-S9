# ImageNet Training with PyTorch Lightning

## About the Project
This project implements a distributed training pipeline for ResNet50 on ImageNet using PyTorch Lightning. The implementation focuses on efficient multi-GPU training with automatic learning rate finding and various optimizations for training speed and stability.

### Key Features
- Multi-GPU training using DistributedDataParallel (DDP)
- Automatic learning rate finding
- Mixed precision training (FP16)
- Gradient clipping for training stability
- TensorBoard logging and model checkpointing
- Support for both ReduceLROnPlateau and StepLR schedulers

## Training Infrastructure

### Minimum Hardware Requirements
- 4x NVIDIA GPUs
- CPU: Recommended 16+ cores
- RAM: 32GB+ recommended
- Storage: SSD with sufficient space for ImageNet dataset

### Actual Hardware used for training
- g5.12xlarge (4x A10G)
- 48 vCPU
- 192GB RAM
- 500GB SSD

### Software Requirements
- Python 3.10
- PyTorch 2.0
- PyTorch Lightning 2.0
- CUDA 12.0
- cuDNN 8.9
- NVIDIA Container Toolkit

## Performance Optimization Tips
1. **Data Loading**:
   - Adjust number of workers based on CPU cores
   - Use appropriate batch size for GPU memory
   - Enable persistent workers for faster epoch starts

2. **GPU Utilization**:
   - Monitor GPU usage with `nvidia-smi`
   - Adjust batch size if GPUs are underutilized
   - Use mixed precision training for better performance

3. **Memory Management**:
   - Use gradient clipping to prevent memory spikes
   - Enable automatic mixed precision (AMP)
   - Monitor memory usage during training

## License
This project is licensed under the MIT License - see the LICENSE file for details. 

### Training Configuration
- **Model**: ResNet50
- **Batch Size**: 32 per GPU
- **Initial Learning Rate**: Determined automatically by LR finder
- **Optimizer**: SGD with momentum (0.9)
- **Weight Decay**: 1e-4
- **Training Duration**: 100 epochs
- **Precision**: Mixed precision (FP16)

### Learning Rate Scheduling
Two options available (controlled by `USE_REDUCE_LR` flag):
1. **ReduceLROnPlateau**:
   - Factor: 0.1
   - Patience: 10 epochs
   - Minimum LR: 1e-6

2. **StepLR**:
   - Step Size: 30 epochs
   - Gamma: 0.1

### Data Augmentation
Training transforms:
- Random resized crop (224x224)
- Random horizontal flip
- Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Validation transforms:
- Resize to 256x256
- Center crop to 224x224
- Normalization (same as training)

## Training Process

### Setup and Training
1. Install dependencies: 
## Training Logs

### Learning Rate Finder
The learning rate finder automatically determines the optimal learning rate before training begins. Results are saved as 'lr_finder_plot.png'.

## Markdown Logs of training
View detailed training logs and visualizations in [viewlogs.md](viewlogs.md). Training visualizations include:
- Learning rate finder plots
- Training metrics over time
- GPU utilization graphs

All visualization images are stored in the `images/` directory. 

### Dataset Structure 
data/
├── train/
│ ├── class1/
│ │ ├── image1.JPEG
│ │ └── ...
│ └── class2/
└── val/
├── class1/
│ ├── image1.JPEG
│ └── ...
└── class2/


## Model Deployment

### Saving Checkpoints
The model automatically saves checkpoints during training:
- Location: `checkpoints/{model_name}`
- Format: `{epoch}-{val_loss:.2f}.ckpt`
- Keeps top 3 models based on validation loss

