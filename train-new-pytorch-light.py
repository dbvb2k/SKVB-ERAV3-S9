import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import time
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
import csv
from datetime import datetime

# Import PyTorch and Lightning libraries
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.tuner.tuning import Tuner

# Constants
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_DASHES = 110
USE_REDUCE_LR = True
NUM_WORKERS = 0
GRADIENT_CLIP_VAL = 0.1

class Params:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.name = "resnet_50_sgd"
        self.workers = 0
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        # LR Finder parameters
        self.min_lr = 1e-8
        self.max_lr = 0.01
        self.num_training = 100

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class ResNet50LightningModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        
        # Initialize model with better defaults
        self.model = models.resnet50(weights=None)
        
        # Initialize weights with smaller values
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
        
        self.criterion = nn.CrossEntropyLoss()
        self.automatic_lr = None
        self.learning_rate = params.lr
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Update autocast usage
        with torch.amp.autocast(device_type='cuda'):
            logits = self(x)
            loss = self.criterion(logits, y)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}")
            raise ValueError("NaN loss detected")
        
        # Add warmup for first epoch
        if self.trainer.current_epoch == 0:
            warmup_steps = 100
            if self.trainer.global_step < warmup_steps:
                lr_scale = min(1., float(self.trainer.global_step + 1) / warmup_steps)
                for pg in self.optimizers().param_groups:
                    pg['lr'] = lr_scale * self.learning_rate
        
        # Ensure all gradients are synchronized
        if self.trainer.strategy.launcher is not None:
            torch.distributed.barrier()
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Calculate top-5 accuracy
        _, pred_top5 = logits.topk(5, 1, largest=True, sorted=True)
        correct_top5 = pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).float().sum(dim=1).mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc * 100, prog_bar=True)
        self.log('val_acc_top5', correct_top5 * 100, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=True
        )
        
        if USE_REDUCE_LR:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    min_lr=1e-6
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        else:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=NUM_EPOCHS,
                    eta_min=1e-6
                ),
                'interval': 'epoch',
                'frequency': 1
            }
        
        return [optimizer], [scheduler]

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = 0  # Start with no workers for debugging
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        try:
            if stage == 'fit' or stage is None:
                if not os.path.exists(self.train_dir):
                    raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
                if not os.path.exists(self.val_dir):
                    raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")
                
                try:
                    self.train_dataset = torchvision.datasets.ImageFolder(
                        root=self.train_dir,
                        transform=self.train_transform
                    )
                    self.val_dataset = torchvision.datasets.ImageFolder(
                        root=self.val_dir,
                        transform=self.val_transform
                    )
                except Exception as e:
                    print(f"Error loading datasets: {str(e)}")
                    raise
                
                # Verify datasets are loaded correctly
                if len(self.train_dataset) == 0:
                    raise ValueError("Training dataset is empty")
                if len(self.val_dataset) == 0:
                    raise ValueError("Validation dataset is empty")
                
                if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                    print(f"Found {len(self.train_dataset)} training images")
                    print(f"Found {len(self.val_dataset)} validation images")
                    
                    # Test loading a few batches
                    try:
                        train_loader = self.train_dataloader()
                        for i, (images, labels) in enumerate(train_loader):
                            if i >= 2:  # Test first 3 batches
                                break
                            print(f"Successfully loaded batch {i+1}")
                    except Exception as e:
                        print(f"Error testing data loading: {str(e)}")
                        raise
        except Exception as e:
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f"Error in setup: {str(e)}")
            raise
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # Disable pin_memory
            persistent_workers=False,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,  # Disable pin_memory
            persistent_workers=False
        )

def print_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            print(f"GPU {i} - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

class TrainingEpochLogger(Callback):
    def __init__(self, log_dir="training_logs"):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = None
        self.total_epochs = None
        
    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:  # Only create log file on main process
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, f"training_log_{timestamp}.txt")
            self.log_file = open(log_path, 'w')
            self.total_epochs = trainer.max_epochs
            
            # Write header
            self.log_file.write("ImageNet Training with ResNet50\n")
            self.log_file.write(f"Started at: {timestamp}\n")
            self.log_file.write("=" * 180 + "\n\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Only log on main process
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch
            
            # Get current learning rate
            if trainer.optimizers:
                current_lr = trainer.optimizers[0].param_groups[0]['lr']
            else:
                current_lr = pl_module.learning_rate
            
            # Get metrics (with default values if not found)
            train_loss = metrics.get('train_loss', float('nan'))
            val_loss = metrics.get('val_loss', float('nan'))
            val_acc = metrics.get('val_acc', float('nan'))
            val_acc_top5 = metrics.get('val_acc_top5', float('nan'))
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format the log message
            log_message = (
                f"Epoch {epoch + 1}/{self.total_epochs} - "
                f"Training Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_acc:.2f}%, "
                f"Validation Top-5 Accuracy: {val_acc_top5:.2f}%, "
                f"Learning Rate: {current_lr:.8f}, "
                f"Timestamp: {timestamp}"
            )
            
            # Write to file
            self.log_file.write(log_message + "\n")
            self.log_file.flush()  # Ensure writing to disk
            
            # Also print to console
            print(log_message)
    
    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero and self.log_file:  # Close file on main process
            # Write footer
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write("\n" + "=" * 180 + "\n")
            self.log_file.write(f"Training completed at: {end_timestamp}\n")
            self.log_file.close()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero and batch_idx % 100 == 0:
            print_memory_stats()
            # pass

def cleanup():
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except:
        pass
    finally:
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def main():
    # Memory management
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Add Tensor Core optimization
    torch.set_float32_matmul_precision('high')
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main_process = rank == 0
    
    if is_main_process:
        print("\n" + "=" * NUM_DASHES)
        print("INITIALIZING TRAINING PIPELINE")
        print("=" * NUM_DASHES)
    
    params = Params()
    training_folder_name = './data/train'
    val_folder_name = './data/val'
    
    try:
        data_module = ImageNetDataModule(training_folder_name, val_folder_name)
        model = ResNet50LightningModule(params)
        
        # Setup callbacks and logger for main process
        callbacks = []
        logger = False
        
        if is_main_process:
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"checkpoints/{params.name}",
                filename="{epoch}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            )
            lr_monitor = LearningRateMonitor(logging_interval='step')
            epoch_logger = TrainingEpochLogger()
            callbacks = [checkpoint_callback, lr_monitor, epoch_logger]
            logger = TensorBoardLogger("lightning_logs", name=params.name)
            
            # Run learning rate finder only on main process
            lr_finder_trainer = pl.Trainer(
                accelerator="gpu",
                devices=[0],  # Explicitly use first GPU
                logger=False,
                enable_progress_bar=True,
            )
            
            tuner = Tuner(lr_finder_trainer)
            print("\nRunning learning rate finder...")
            
            # Run LR finder with a clean model instance
            lr_finder_model = ResNet50LightningModule(params)
            lr_finder = tuner.lr_find(
                lr_finder_model,
                data_module,
                min_lr=params.min_lr,
                max_lr=params.max_lr,
                num_training=params.num_training,
                attr_name="learning_rate",
            )
            
            if lr_finder.suggestion() is not None:
                new_lr = lr_finder.suggestion()
                model.automatic_lr = new_lr
                model.learning_rate = new_lr
                print(f"Learning rate finder suggests: {new_lr:.8f}")
                
                fig = lr_finder.plot(suggest=True)
                fig.savefig('lr_finder_plot.png')
                print("Learning rate finder plot saved as 'lr_finder_plot.png'")
            else:
                new_lr = params.lr
                model.automatic_lr = new_lr
                model.learning_rate = new_lr
                print("Using default learning rate:", new_lr)
        
        # Synchronize the learning rate across processes
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if is_main_process:
                torch.distributed.broadcast_object_list([model.learning_rate], src=0)
            else:
                lr_container = [None]
                torch.distributed.broadcast_object_list(lr_container, src=0)
                model.learning_rate = lr_container[0]
                model.automatic_lr = lr_container[0]
        
        # Initialize trainer with modified settings
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="gpu",
            devices=4,
            strategy="ddp",  # Simplified DDP strategy
            callbacks=callbacks,
            logger=logger,
            precision="16-mixed",
            log_every_n_steps=100,
            enable_progress_bar=is_main_process,
            deterministic=False,
            benchmark=True,
            sync_batchnorm=True,
            gradient_clip_val=GRADIENT_CLIP_VAL,
            gradient_clip_algorithm="norm",
            detect_anomaly=False,
            num_sanity_val_steps=2,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=2
        )
        
        if is_main_process:
            print(f"\nStarting training with learning rate: {model.learning_rate:.8f}")
        
        trainer.fit(model, data_module)
        
    except Exception as e:
        if is_main_process:
            print(f"Error during training setup: {str(e)}")
        raise
    finally:
        cleanup()

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    try:
        main()
    finally:
        cleanup()





