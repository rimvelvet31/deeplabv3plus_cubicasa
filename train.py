import os
import time
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import RandomChoice
from torchmetrics import MetricCollection, Accuracy, JaccardIndex

from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (Compose,
                                              RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              RandomRotations,
                                              DictToTensor,
                                              ColorJitterTorch)

from model.deeplabv3plus import DeepLabV3Plus
from loss_fn import calculate_class_weights, MultiTaskUncertaintyLoss

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Set seed for reproducibility
torch.manual_seed(1)

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using device: CPU')

# Clear cache to prevent CUDA out of memory errors
torch.cuda.empty_cache()

# Parse command line arguments for setting hyperparameters
parser = argparse.ArgumentParser(description='Train DeepLabV3+ on the CubiCasa5k dataset')
parser.add_argument('--img_size', type=int, default=256, help='Size to resize the images (default: 256)')
parser.add_argument('--bs', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (default: 0)')
parser.add_argument('--backbone', type=str, default='mobilenet_v2', help='Backbone for DeepLabV3+ (default: mobilenet_v2)')
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
    # RandomChoice([
    #     RandomCropToSizeTorch(data_format='dict', size=IMAGE_SIZE),
    #     ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE)
    # ]),
    ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE),
    RandomRotations(format='cubi'),
    DictToTensor(),
    ColorJitterTorch(b_var=0.1, c_var=0.1, s_var=0.1)
])

val_aug = Compose([
    ResizePaddedTorch((0, 0), data_format='dict', size=IMAGE_SIZE),
    DictToTensor()
])

# Dataset and DataLoaders
data_path = 'data/cubicasa5k/'
format = 'lmdb'

train_set = FloorplanSVG(data_path, 'train.txt', format=format, augmentations=train_aug)
val_set = FloorplanSVG(data_path, 'val.txt', format=format, augmentations=val_aug)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

# Model Setup
model = DeepLabV3Plus(backbone=BACKBONE, attention=USE_ATTENTION).to(device)

if USE_ATTENTION:
    print(f'Model: DeepLabV3+ with {BACKBONE} backbone and CA + SA modules\n')
else:
    print(f'Model: DeepLabV3+ with {BACKBONE} backbone and no attention modules\n')

# Loss functions
# room_class_weights = calculate_class_weights(train_loader, num_classes=12, label_index=21)
# icon_class_weights = calculate_class_weights(train_loader, num_classes=11, label_index=22)
# print(f'Room class weights: {room_class_weights}')
# print(f'Icon class weights: {icon_class_weights}\n')

# Computed from earlier to avoid recomputing on every run
room_class_weights = torch.tensor([0.0020, 0.0290, 0.0175, 0.0297, 0.0168, 0.0191, 
                                   0.0627, 0.0437, 0.4499,0.1118, 0.2035, 0.0144]).to(device)
icon_class_weights = torch.tensor([4.6019e-05, 4.1143e-03, 9.0201e-03, 3.1095e-03, 6.7754e-03, 2.8754e-02,
                                   1.6963e-02, 9.4359e-03, 6.6047e-02, 1.0371e-01, 7.5203e-01]).to(device)

room_criterion = nn.CrossEntropyLoss(weight=room_class_weights)
icon_criterion = nn.CrossEntropyLoss(weight=icon_class_weights)
heatmap_criterion = nn.MSELoss()
multitask_criterion = MultiTaskUncertaintyLoss()

# These are based on CubiCasa5k training setup
# Weight decay to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

# Load best model checkpoint if available or train from scratch
checkpoint_path = f'saved_models/dlv3p_{model.backbone_name}_{model.attention}.pt'

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


def train_epoch(model, dataloader, optimizer, device):
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
        multitask_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

        # Backward pass and optimization
        multitask_loss.backward()
        optimizer.step()

        total_loss += multitask_loss.item()

    epoch_loss = total_loss / len(dataloader)
    return epoch_loss


def validate_epoch(model, dataloader, device):
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    FILE_NAME = f"deeplab_{model.backbone_name}_{model.attention}_{timestamp}"

    # Setup logging
    os.makedirs('logs', exist_ok=True)

    log_filename = f"{FILE_NAME}.log"
    log_filepath = os.path.join('logs', log_filename)

    logging.basicConfig(filename=log_filepath, level=logging.INFO,format='%(asctime)s - %(message)s')
    logging.info(f'Logging training for DeepLabV3+ with {BACKBONE} backbone and attention: {USE_ATTENTION}\n')

    # TRAINING AND VALIDATION LOOP
    train_loss_history = []
    val_loss_history = []
    room_mpa_history = []
    room_miou_history = []
    room_fwiou_history = []
    icon_mpa_history = []
    icon_miou_history = []
    icon_fwiou_history = []

    lr_dropped = False
    early_stopping_counter = 0
    early_stopping_patience = 30

    for epoch in range(start_epoch, EPOCHS):
        print(f"EPOCH {epoch+1}/{EPOCHS}")

        # Get loss and metrics
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_room_metrics, val_icon_metrics = validate_epoch(model, val_loader, device)

        # Get actual values for each metric
        room_mpa = val_room_metrics['mpa'].item()
        room_miou = val_room_metrics['miou'].item()
        room_fwiou = val_room_metrics['fwiou'].item()

        icon_mpa = val_icon_metrics['mpa'].item()
        icon_miou = val_icon_metrics['miou'].item()
        icon_fwiou = val_icon_metrics['fwiou'].item()

        # Save metrics to history for plotting
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        room_mpa_history.append(room_mpa)
        room_miou_history.append(room_miou)
        room_fwiou_history.append(room_fwiou)
        icon_mpa_history.append(icon_mpa)
        icon_miou_history.append(icon_miou)
        icon_fwiou_history.append(icon_fwiou)

        # Log results for this epoch
        log_message = (
            f'Train Loss: {train_loss:.4f}\n'
            f'Val Loss:   {val_loss:.4f}\n'
            f'Rooms - mPA: {room_mpa:.4f}, mIoU: {room_miou:.4f}, fwIoU: {room_fwiou:.4f}\n'
            f'Icons - mPA: {icon_mpa:.4f}, mIoU: {icon_miou:.4f}, fwIoU: {icon_fwiou:.4f}\n'
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
            early_stop_msg = f'\nEarly stopping triggered after {epoch + 1} epochs'
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

    # Visualize training and validation results and save to PDF
    os.makedirs('visualizations', exist_ok=True)
    pdf_filepath = os.path.join('visualizations', f"{FILE_NAME}.pdf")

    with PdfPages(pdf_filepath) as pdf:
        # Training vs Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Room vs Icon mPA
        plt.figure(figsize=(10, 6))
        plt.plot(room_mpa_history, label='Room mPA')
        plt.plot(icon_mpa_history, label='Icon mPA')
        plt.title('Room vs Icon Mean Pixel Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('mPA')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Room vs Icon mIoU
        plt.figure(figsize=(10, 6))
        plt.plot(room_miou_history, label='Room mIoU')
        plt.plot(icon_miou_history, label='Icon mIoU')
        plt.title('Room vs Icon Mean Intersection over Union')
        plt.xlabel('Epochs')
        plt.ylabel('Mean IoU')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Room vs Icon fwIoU
        plt.figure(figsize=(10, 6))
        plt.plot(room_fwiou_history, label='Room fwIoU')
        plt.plot(icon_fwiou_history, label='Icon fwIoU')
        plt.title('Room vs Icon Frequency-Weighted IoU')
        plt.xlabel('Epochs')
        plt.ylabel('fwIoU')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

    print('Training completed successfully!')
