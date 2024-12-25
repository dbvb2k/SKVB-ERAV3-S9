import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from tqdm import tqdm

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Constants
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64  # Separate constant for evaluation batch size
NUM_EPOCHS = 100
NUM_DASHES = 110
USE_REDUCE_LR = True  # Flag to choose between schedulers

""" 
# If this is needed, uncomment the following code and also uncomment the import of matplotlib.pyplot as plt

def show_image(image, label):
    image = image.permute(1, 2, 0)
    plt.imshow(image.squeeze())
    plt.title(f'Label: {label}')
    plt.show() """

class Params:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.name = "resnet_50_sgd"
        self.workers = 4
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        # StepLR parameters
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        # ReduceLROnPlateau parameters
        self.lr_patience = 10
        self.lr_factor = 0.1
        self.lr_min = 1e-6

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def get_best_training_params():
    params = {
        'batch_size': BATCH_SIZE,  # Use the constant here
        'lr': 0.008760072654821956,
        'momentum': 0.9183217735548175,
        'weight_decay': 0.0004041898838198785,
        'max_lr': 0.25950600108516736,
        'pct_start': 0.2883974370969041,
        'div_factor': 15,
        'final_div_factor': 200
    }
    return params

def get_resnet50_model():
    model = models.resnet50(weights=None)
    training_params = get_best_training_params()
    return model

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    
    print(f"\nEpoch {epoch+1}")
    print("-" * NUM_DASHES)
    
    # Create progress bar for the epoch
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}', 
                unit='batch', 
                total=len(dataloader),
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    running_loss = 0.0
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update running loss and progress bar
        running_loss += loss.item()
        avg_loss = running_loss / (batch + 1)
        
        # Update progress bar description
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'progress': f'{((batch+1)*len(X))/size*100:.1f}%'
        })
        
        # Log to tensorboard every 100 batches
        if batch % 100 == 0:
            step = epoch * size + batch * len(X)
            writer.add_scalar('Training loss', loss.item(), step)
    
    epoch_time = time.time() - start0
    print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds")
    print(f"Average loss: {avg_loss:.4f}")
    print("-" * NUM_DASHES)

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    print(f"\nEvaluating epoch {epoch+1}")
    print("-" * NUM_DASHES)
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0
    
    # Create progress bar for evaluation
    pbar = tqdm(dataloader, desc=f'Evaluating Epoch {epoch+1}',
                unit='batch',
                total=len(dataloader),
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            # Update progress bar with current metrics
            avg_loss = test_loss / (pbar.n + 1)
            current_acc = (correct / ((pbar.n + 1) * X.size(0))) * 100
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{current_acc:.1f}%'
            })
    
    test_loss /= num_batches
    correct /= size
    correct_top5 /= size if calc_acc5 else 1
    
    # Log metrics to tensorboard
    step = epoch * len(train_dataloader.dataset)
    if writer is not None:
        writer.add_scalar('Test loss', test_loss, step)
        writer.add_scalar('Test accuracy', 100 * correct, step)
        if calc_acc5:
            writer.add_scalar('Test accuracy5', 100 * correct_top5, step)
    
    print(f"\nTest Results - Epoch: {epoch+1}")
    print(f"Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f}")
    if calc_acc5:
        print(f"Top-5 Accuracy: {(100*correct_top5):>0.1f}%")
    print("-" * NUM_DASHES)
    
    return test_loss

def main():
    print("\n" + "=" * NUM_DASHES)
    print("INITIALIZING TRAINING PIPELINE")
    print("=" * NUM_DASHES)
    
    global device  # We'll need this in other functions
    
    # Device setup
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n[INFO] Using {device} device")

    # Parameters setup
    print("\n[SETUP] Initializing training parameters...")
    params = Params()
    resume_training = True
    print(f"[SETUP] Training parameters initialized: \n{params}")
    
    # Data paths
    print("\n[DATA] Setting up data paths...")
    training_folder_name = './data/Imagenet-Mini/train'
    val_folder_name = './data/Imagenet-Mini/val'
    print(f"[DATA] Training data path: {training_folder_name}")
    print(f"[DATA] Validation data path: {val_folder_name}")
    
    # Data transformations and loaders
    print("\n[AUGMENTATION] Setting up data transformations...")
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[DATA] Loading training dataset...")
    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    print(f"[DATA] Training dataset size: {len(train_dataset)} images")
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # Use the constant here
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    print(f"[DATA] Training batches: {len(train_loader)}")

    print("\n[DATA] Loading validation dataset...")
    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )
    print(f"[DATA] Validation dataset size: {len(val_dataset)} images")
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,  # Use the evaluation batch size constant
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )
    print(f"[DATA] Validation batches: {len(val_loader)}")

    # Model setup
    print("\n[MODEL] Setting up ResNet50 model...")
    model = get_resnet50_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=params.lr, 
                               momentum=params.momentum, 
                               weight_decay=params.weight_decay)
    print("[MODEL] Model initialized and moved to device")

    # Resume training setup
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    if resume_training and os.path.exists(checkpoint_path):
        print("\n[CHECKPOINT] Loading previous checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        
        try:
            assert params == checkpoint["params"]
            # Load checkpoint only if parameters match
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"[CHECKPOINT] Resuming training from epoch {start_epoch}")
        except AssertionError:
            print("[WARNING] Checkpoint parameters don't match current parameters.")
            print("[WARNING] Starting fresh training...")
            start_epoch = 0
    else:
        print("\n[CHECKPOINT] No checkpoint found, starting fresh training")

    # Training setup
    print("\n[SETUP] Creating checkpoint directory...")
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    print("[SETUP] TensorBoard writer initialized")
    
    print("\n[EVALUATION] Running initial model evaluation...")
    # Initial test
    test(val_loader, model, loss_fn, epoch=-1, writer=writer, train_dataloader=train_loader, calc_acc5=True)

    # print("\n" + "=" * NUM_DASHES)
    print("\nSTARTING TRAINING LOOP")
    print(f"Training for {NUM_EPOCHS-start_epoch} epochs")
    print("=" * NUM_DASHES + "\n")

    # Learning rate scheduler setup
    print("[SCHEDULER] Setting up learning rate scheduler...")
    if USE_REDUCE_LR:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.lr_factor,
            patience=params.lr_patience,
            verbose=True,
            min_lr=params.lr_min
        )
        print("[SCHEDULER] Using ReduceLROnPlateau scheduler")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.lr_step_size,
            gamma=params.lr_gamma
        )
        print("[SCHEDULER] Using StepLR scheduler")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer)
        
        # Evaluate the model
        test_loss = test(val_loader, model, loss_fn, epoch, writer, train_dataloader=train_loader, calc_acc5=True)
        
        # Step the scheduler
        if USE_REDUCE_LR:
            lr_scheduler.step(test_loss)  # Use validation loss for ReduceLROnPlateau
        else:
            lr_scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        print(f"\n[CHECKPOINT] Saving checkpoint for epoch {epoch+1}...")
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()





