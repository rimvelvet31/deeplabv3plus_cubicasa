# General
import os
import time
import argparse
import logging
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

# Own
from model.deeplabv3plus import DeepLabV3Plus
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


# Parse command line arguments for setting hyperparameters
parser = argparse.ArgumentParser(description='Train DeepLabV3+ on the CubiCasa5k dataset')
parser.add_argument('--img_size', type=int, default=256, help='Size to resize the images (default: 256)')
parser.add_argument('--bs', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (default: 0)')
parser.add_argument('--backbone', type=str, default='mobilenetv2', help='Backbone for DeepLabV3+ (default: mobilenetv2)')
parser.add_argument('--attention', action='store_true', help='Use CA and SA modules (default: False)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
parser.add_argument('--reload_best_model', action='store_true', help='Resume training from best model checkpoint (default: False)')

args = parser.parse_args()

IMAGE_SIZE = (args.img_size, args.img_size)
BATCH_SIZE = args.bs
NUM_WORKERS = args.workers
BACKBONE = args.backbone
USE_ATTENTION = args.attention
EPOCHS = args.epochs
INITIAL_LR = args.lr
RELOAD_BEST_MODEL = args.reload_best_model


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
data_path = 'data/cubicasa5k/'
format = 'lmdb'

train_set = FloorplanSVG(
    data_path, 
    'train.txt', 
    format=format, 
    augmentations=train_aug
)

# Reduce training set for faster training (temporary)
# train_set = Subset(full_train_set, list(range(2100)))

val_set = FloorplanSVG(
    data_path, 
    'val.txt', 
    format=format, 
    augmentations=val_aug
)

print('Train set size:', len(train_set))
print('Validation set size:', len(val_set))

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

# summary(model, input_size=(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

if USE_ATTENTION:
    print(f'Model Loaded: DeepLabV3+ with {BACKBONE} backbone and CA + SA modules\n')
else:
    print(f'Model Loaded: DeepLabV3+ with {BACKBONE} backbone and no attention modules\n')


# Loss functions
room_criterion = nn.CrossEntropyLoss()
icon_criterion = nn.CrossEntropyLoss()
heatmap_criterion = nn.MSELoss()
multitask_criterion = MultiTaskUncertaintyLoss(num_tasks=3).to(device)

# These are based on CubiCasa5k training setup
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)


# Load best model checkpoint if available or train from scratch
checkpoint_path = f'saved_models/best_checkpoint_{model.backbone_name}_{model.attention}.pt'

if RELOAD_BEST_MODEL and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}\n")
else:
    best_val_loss = float('inf')
    start_epoch = 0
    print("Starting training from scratch\n")


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


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc='Training'):
        images = batch['image'].float().to(device)
        room_labels = batch['label'][:, 21].long().to(device)
        icon_labels = batch['label'][:, 22].long().to(device)
        heatmap_labels = batch['label'][:, 0:21].to(device)

        optimizer.zero_grad()

        # Forward pass
        room_output, icon_output, heatmap_output = model(images)

        # Compute individual losses then feed into multi-task loss
        room_loss = room_criterion(room_output, room_labels)
        icon_loss = icon_criterion(icon_output, icon_labels)
        heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)

        combined_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

        # Backward pass and optimization
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()

        # Get model predictions
        room_preds = torch.argmax(room_output, dim=1)
        icon_preds = torch.argmax(icon_output, dim=1)

        # By batch
        room_metrics.update(room_preds, room_labels)
        icon_metrics.update(icon_preds, icon_labels)

    # By epoch
    epoch_loss = total_loss / len(dataloader)
    epoch_room_metrics = room_metrics.compute()
    epoch_icon_metrics = icon_metrics.compute()

    room_metrics.reset()
    icon_metrics.reset()

    return epoch_loss, epoch_room_metrics, epoch_icon_metrics


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images = batch['image'].float().to(device)
            room_labels = batch['label'][:, 21].long().to(device)
            icon_labels = batch['label'][:, 22].long().to(device)
            heatmap_labels = batch['label'][:, 0:21].to(device)

            room_output, icon_output, heatmap_output = model(images)

            room_loss = room_criterion(room_output, room_labels)
            icon_loss = icon_criterion(icon_output, icon_labels)
            heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)
            combined_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

            total_loss += combined_loss.item()

            room_preds = torch.argmax(room_output, dim=1)
            icon_preds = torch.argmax(icon_output, dim=1)

            room_metrics.update(room_preds, room_labels)
            icon_metrics.update(icon_preds, icon_labels)
    
    epoch_loss = total_loss / len(dataloader)
    epoch_room_metrics = room_metrics.compute()
    epoch_icon_metrics = icon_metrics.compute()

    room_metrics.reset()
    icon_metrics.reset()
    
    return epoch_loss, epoch_room_metrics, epoch_icon_metrics


if __name__ == '__main__':
    # Setup logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"{model.backbone_name}_{model.attention}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(filename=log_filepath, level=logging.INFO,format='%(asctime)s - %(message)s')
    logging.info(f'Logging training for DeepLabV3+ with {BACKBONE} backbone and attention: {USE_ATTENTION}\n')

    # Setup for training and validation
    lr_dropped = False
    early_stopping_patience = 30
    early_stopping_counter = 0

    for epoch in range(start_epoch, EPOCHS):
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

        log_message = (
            f"\nTRAIN RESULTS\n"
            f"Loss: {train_loss:.4f}\n"
            f"Rooms - mPA: {train_room_mpa:.4f}, mIoU: {train_room_miou:.4f}, fwIoU: {train_room_fwiou:.4f}\n"
            f"Icons - mPA: {train_icon_mpa:.4f}, mIoU: {train_icon_miou:.4f}, fwIoU: {train_icon_fwiou:.4f}\n"
            f"\nVALIDATION RESULTS\n"
            f"Loss: {val_loss:.4f}\n"
            f"Rooms - mPA: {val_room_mpa:.4f}, mIoU: {val_room_miou:.4f}, fwIoU: {val_room_fwiou:.4f}\n"
            f"Icons - mPA: {val_icon_mpa:.4f}, mIoU: {val_icon_miou:.4f}, fwIoU: {val_icon_fwiou:.4f}\n"
        )
        logging.info(log_message)
        print(log_message)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)

            save_msg = f'\nBest model saved with validation loss: {best_val_loss:.4f}'
            print(save_msg)
            logging.info(save_msg)

        else:
            early_stopping_counter += 1
            not_improved_msg = f'\nValidation loss did not improve. Best loss is still: {best_val_loss:.4f}'
            print(not_improved_msg)
            logging.info(not_improved_msg)

        # Activate early stopping if no improvement for specific epochs
        if early_stopping_counter >= early_stopping_patience:
            early_stop_msg = f'\nEarly stopping triggered after {epoch+1} epochs'
            print(early_stop_msg)
            logging.info(early_stop_msg)
            break

        # Check if LR was dropped
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Reload best model weights if LR was dropped
        if current_lr < prev_lr and not lr_dropped:
            lr_dropped = True
            model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
            torch.cuda.empty_cache()

            reload_best_msg = f'\nLR dropped to {current_lr}. Model weights reloaded with best validation loss: {best_val_loss:.4f}'
            print(reload_best_msg)
            logging.info(reload_best_msg)

        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
