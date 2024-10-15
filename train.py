# General
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchinfo import summary
from torchmetrics import MetricCollection, Accuracy, JaccardIndex

# CubiCasa
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)

# Model
from model.deeplabv3plus import DeepLabV3Plus

# Multi-task Uncertainty Loss
from loss_fn import MultiTaskUncertaintyLoss

# Release GPU memory
torch.cuda.empty_cache()

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using device: CPU')

# Hyperparameters
IMAGE_SIZE = (256, 256)

DATA_PATH = 'data/cubicasa5k/'
TRAIN_PATH = 'train.txt'
VAL_PATH = 'val.txt'
FORMAT = 'lmdb'

BATCH_SIZE = 32
NUM_WORKERS = 4

BACKBONE = 'mobilenetv2'
USE_ATTENTION = True

EPOCHS = 300

# Preprocessing and Augmentations
train_aug = Compose([
    # transforms.RandomChoice([
    #     RandomCropToSizeTorch(data_format='dict', size=IMAGE_SIZE),
    #     ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE)
    # ]),
    ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE),
    RandomRotations(format='cubi'),
    DictToTensor(),
    ColorJitterTorch(b_var=0.2, c_var=0.2, s_var=0.2)
])

val_aug = Compose([
    ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE),
    DictToTensor()
])

# Dataset and DataLoaders
train_set = FloorplanSVG(
    DATA_PATH, 
    TRAIN_PATH, 
    format=FORMAT, 
    augmentations=train_aug
)

# Reduce training set for faster training (temporary)
# train_set = Subset(full_train_set, list(range(1600)))

val_set = FloorplanSVG(
    DATA_PATH, 
    VAL_PATH, 
    format=FORMAT, 
    augmentations=val_aug
)

# print('Train set size:', len(train_set))
# print('Validation set size:', len(val_set))

train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS, 
    shuffle=True, 
    pin_memory=True
)

val_loader = DataLoader(
    val_set, 
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True
)

# print(f'Length of train dataloader: {len(train_loader)} batches of size {BATCH_SIZE}')
# print(f'Length of val dataloader: {len(val_loader)} batches of size {BATCH_SIZE}')

# batch_sample = next(iter(train_loader))
# print('\nBatch image shape: ', batch_sample['image'].shape)
# print('Batch label shape: ', batch_sample['label'].shape)

# Model Setup
model = DeepLabV3Plus(backbone=BACKBONE, attention=USE_ATTENTION).to(device)

if USE_ATTENTION:
    print(f'Model Loaded: DeepLabV3+ with {BACKBONE} backbone and CA + SA modules\n')
else:
    print(f'Model Loaded: DeepLabV3+ with {BACKBONE} backbone and no attention modules\n')

# summary(model, input_size=(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

# TRAINING AND VALIDATION LOOP
# Individual loss functions for each task
room_criterion = nn.CrossEntropyLoss()
icon_criterion = nn.CrossEntropyLoss()
heatmap_criterion = nn.MSELoss()

# Multi-task uncertainty loss
multitask_criterion = MultiTaskUncertaintyLoss(num_tasks=3).to(device)

# Based on CubiCasa5k training setup
initial_lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)

# Reduce learning rate if validation loss doesnt improve for 20 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

# Evaluation metrics
room_metrics = MetricCollection({
    'mpa': Accuracy(task='multiclass', num_classes=12, average='macro'),
    'miou': JaccardIndex(task='multiclass', num_classes=12, average='macro'),
    'fwiou': JaccardIndex(task='multiclass', num_classes=12, average='weighted'),
}).to(device)

icon_metrics = MetricCollection({
    'mpa': Accuracy(task='multiclass', num_classes=11, average='macro'),
    'miou': JaccardIndex(task='multiclass', num_classes=11, average='macro'),
    'fwiou': JaccardIndex(task='multiclass', num_classes=11, average='weighted'),
}).to(device)


# Training loop
def train(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0

    # To show progress bar
    for batch in tqdm(dataloader, desc='Training'):
        images = batch['image'].float().to(device)
        room_labels = batch['label'][:, 21].long().to(device)
        icon_labels = batch['label'][:, 22].long().to(device)
        heatmap_labels = batch['label'][:, 0:21].to(device)

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        room_output, icon_output, heatmap_output = model(images)

        # Compute individual losses per task
        room_loss = room_criterion(room_output, room_labels)
        icon_loss = icon_criterion(icon_output, icon_labels)
        heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)

        # Compute combined loss
        combined_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

        # Backward pass and optimization
        combined_loss.backward()
        optimizer.step()

        # Accumulate the scalar value of the loss
        total_loss += combined_loss.item()

        # Get model predictions
        room_preds = torch.argmax(room_output, dim=1)
        icon_preds = torch.argmax(icon_output, dim=1)

        # Update metrics
        room_metrics.update(room_preds, room_labels)
        icon_metrics.update(icon_preds, icon_labels)

    # Compute epoch loss and metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_room_metrics = room_metrics.compute()
    epoch_icon_metrics = icon_metrics.compute()

    # Reset metrics
    room_metrics.reset()
    icon_metrics.reset()

    return epoch_loss, epoch_room_metrics, epoch_icon_metrics


# Validation loop
def validate(model, dataloader, device):
    model.eval()

    total_loss = 0.0

    # Don't compute gradients
    with torch.no_grad():

        # Wrap dataloader with tqdm to show progress bar
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].float().to(device)

            room_labels = batch['label'][:, 21].long().to(device)
            icon_labels = batch['label'][:, 22].long().to(device)
            heatmap_labels = batch['label'][:, 0:21].to(device)

            # Forward pass
            room_output, icon_output, heatmap_output = model(images)

            # Compute individual losses
            room_loss = room_criterion(room_output, room_labels)
            icon_loss = icon_criterion(icon_output, icon_labels)
            heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)

            # Compute combined loss
            combined_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

            # Accumulate the scalar value of the loss
            total_loss += combined_loss.item()

            # Get model predictions
            room_preds = torch.argmax(room_output, dim=1)
            icon_preds = torch.argmax(icon_output, dim=1)

            # Update metrics
            room_metrics.update(room_preds, room_labels)
            icon_metrics.update(icon_preds, icon_labels)
    
    # Compute epoch loss and metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_room_metrics = room_metrics.compute()
    epoch_icon_metrics = icon_metrics.compute()

    # Reset metrics
    room_metrics.reset()
    icon_metrics.reset()
    
    return epoch_loss, epoch_room_metrics, epoch_icon_metrics


# Train and validate model
best_val_loss = float('inf')
lr_dropped = False
early_stopping_patience = 30
early_stopping_counter = 0
model_path = f'saved_models/deeplabv3plus_{model.backbone_name}_{model.attention}.pt'

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch+1}/{EPOCHS}")

    train_loss, train_room_metrics, train_icon_metrics = train(model, train_loader, optimizer, device)
    val_loss, val_room_metrics, val_icon_metrics = validate(model, val_loader, device)

    train_room_mpa = train_room_metrics['mpa'].item()
    train_room_miou = train_room_metrics['miou'].item()
    train_room_fwiou = train_room_metrics['fwiou'].item()

    train_icon_mpa = train_icon_metrics['mpa'].item()
    train_icon_miou = train_icon_metrics['miou'].item()
    train_icon_fwiou = train_icon_metrics['fwiou'].item()

    val_room_mpa = val_room_metrics['mpa'].item()
    val_room_miou = val_room_metrics['miou'].item()
    val_room_fwiou = val_room_metrics['fwiou'].item()

    val_icon_mpa = val_icon_metrics['mpa'].item()
    val_icon_miou = val_icon_metrics['miou'].item()
    val_icon_fwiou = val_icon_metrics['fwiou'].item()

    print('\nTRAIN RESULTS')
    print(f'Loss: {train_loss:.4f}')
    print(f'Rooms - mPA: {train_room_mpa:.4f}, mIoU: {train_room_miou:.4f}, fwIoU: {train_room_fwiou:.4f}')
    print(f'Icons - mPA: {train_icon_mpa:.4f}, mIoU: {train_icon_miou:.4f}, fwIoU: {train_icon_fwiou:.4f}')

    print('\nVALIDATION RESULTS')
    print(f'Loss: {val_loss:.4f}')
    print(f'Rooms - mPA: {val_room_mpa:.4f}, mIoU: {val_room_miou:.4f}, fwIoU: {val_room_fwiou:.4f}')
    print(f'Icons - mPA: {val_icon_mpa:.4f}, mIoU: {val_icon_miou:.4f}, fwIoU: {val_icon_fwiou:.4f}')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        lr_dropped = False
        early_stopping_counter = 0
        torch.save(model.state_dict(), model_path)
        print(f'\nBest model saved with validation loss: {best_val_loss:.4f}')
    else:
        early_stopping_counter += 1
        print(f'\nValidation loss did not improve.Best loss is still: {best_val_loss:.4f}')

    # Terminate if model didn't improve for the early stopping patience
    if early_stopping_counter >= early_stopping_patience:
        print(f'\nEarly stopping triggered after {epoch+1} epochs')
        break

    # Check if LR was dropped
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Reload best model weights if LR was dropped
    if current_lr < prev_lr and not lr_dropped:
        lr_dropped = True
        model.load_state_dict(torch.load(model_path))
        print(f'\nLR dropped to {current_lr}. Model weights reloaded with best validation loss: {best_val_loss:.4f}')

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')