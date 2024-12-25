import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time

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

def show_image(image, label):
    image = image.permute(1, 2, 0)
    plt.imshow(image.squeeze())
    plt.title(f'Label: {label}')
    plt.show()

class Params:
    def __init__(self):
        self.batch_size = 16
        self.name = "resnet_50_sgd"
        self.workers = 4
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

def get_best_training_params():
    params = {
        'batch_size': 128,
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
    start = time.time()
    print(f"\nEpoch {epoch+1}")
    print("-" * 50)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        batch_size = len(X)
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"Epoch [{epoch+1}/100] - Batch [{batch:>5d}/{len(dataloader):>5d}]")
            print(f"Loss: {loss:>7f}  Progress: [{current:>5d}/{size:>5d}], {(current/size * 100):>4.1f}%")
            step = epoch * size + current
            writer.add_scalar('Training loss', loss, step)
            new_start = time.time()
            delta = new_start - start
            start = new_start
            if batch != 0:
                print(f"Batch time: {delta:.2f} seconds")
                remaining_steps = size - current
                speed = 100 * batch_size / delta
                remaining_time = remaining_steps / speed
                print(f"Estimated time remaining: {remaining_time:.2f} seconds")
                print("-" * 30)
        optimizer.zero_grad()
    
    epoch_time = time.time() - start0
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    print("=" * 50)

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, calc_acc5=False):
    print(f"\nEvaluating epoch {epoch}")
    print("-" * 30)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct_top5 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
    test_loss /= num_batches
    step = epoch * len(train_dataloader.dataset)
    if writer != None:
        writer.add_scalar('Test loss', test_loss, step)
    correct /= size
    correct_top5 /= size
    if writer != None:
        writer.add_scalar('Test accuracy', 100 * correct, step)
        if calc_acc5:
            writer.add_scalar('Test accuracy5', 100 * correct_top5, step)
            
    print(f"Test Results - Epoch: {epoch}")
    print(f"Accuracy: {(100*correct):>0.1f}%")
    print(f"Average loss: {test_loss:>8f}")
    if calc_acc5:
        print(f"Top-5 Accuracy: {(100*correct_top5):>0.1f}%")
    print("-" * 30)

def main():
    global device  # We'll need this in other functions
    
    # Device setup
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Parameters setup
    params = Params()
    resume_training = True
    
    # Data paths
    training_folder_name = './data/imagenet/train'
    val_folder_name = './data/imagenet/val'
    
    # Data transformations and loaders
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Set to 0 to debug multiprocessing issues
        pin_memory=True,
    )

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
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=0,  # Set to 0 to debug multiprocessing issues
        shuffle=False,
        pin_memory=True
    )

    # Model setup
    model = get_resnet50_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=params.lr, 
                               momentum=params.momentum, 
                               weight_decay=params.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=params.lr_step_size, 
                                                  gamma=params.lr_gamma)

    # Resume training setup
    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        assert params == checkpoint["params"]

    # Training setup
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    
    # Initial test
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, calc_acc5=True)

    # Training loop
    for epoch in range(start_epoch, 100):
        train(train_loader, model, loss_fn, optimizer, epoch=epoch, writer=writer)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        lr_scheduler.step()
        
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader, calc_acc5=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()





