import os
import time
import argparse
import logging

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
from train_utils import train_epoch, validate_epoch, visualize
from losses import *

from matplotlib.backends.backend_pdf import PdfPages

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using device: CPU')

# Prevent CUDA out of memory errors when rerunning the script
torch.cuda.empty_cache()

# Get CLI args to set hyperparameters
parser = argparse.ArgumentParser(description='Train DeepLabV3+ on the CubiCasa5k dataset')
parser.add_argument('--img_size', type=int, default=256, help='Size to resize the images (default: 256)')
parser.add_argument('--bs', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (default: 0)')
parser.add_argument('--backbone', type=str, default='mobilenet_v2', help='Backbone for DeepLabV3+ (default: mobilenet_v2)')
parser.add_argument('--attention', action='store_true', help='Use CA and SA modules (default: False)')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train (default: 400)')
parser.add_argument('--loss', type=str, default='dice', help='Loss function (options: dice, weighted_dice, combo)')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
parser.add_argument('--wd', type=float, default=0.01, help='Weight decay (default: 0.01)')
parser.add_argument('--scheduler', type=str, default='onecycle', help='Learning rate scheduler (options: onecycle, cawr)')
parser.add_argument('--reload_best_model', action='store_true', help='Resume training from best model checkpoint (default: False)')
args = parser.parse_args()

IMAGE_SIZE = (args.img_size, args.img_size)
BATCH_SIZE = args.bs
NUM_WORKERS = args.workers
BACKBONE = args.backbone
USE_ATTENTION = args.attention
EPOCHS = args.epochs
LOSS_FN = args.loss
INITIAL_LR = args.lr
WEIGHT_DECAY = args.wd
SCHEDULER = args.scheduler
RELOAD_BEST_MODEL = args.reload_best_model

# Data Preprocessing and Augmentations
train_aug = Compose([
    # RandomChoice([
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
train_set = FloorplanSVG('data/cubicasa5k/', 'train.txt', format='lmdb', augmentations=train_aug)
val_set = FloorplanSVG('data/cubicasa5k/', 'val.txt', format='lmdb', augmentations=val_aug)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

# Model Setup
model = DeepLabV3Plus(backbone=BACKBONE, attention=USE_ATTENTION).to(device)

if USE_ATTENTION:
    model_msg = f'Model: DeepLabV3+ with {BACKBONE} backbone and CA + SA modules\n'
else:
    model_msg = f'Model: DeepLabV3+ with {BACKBONE} backbone and no attention modules\n'

# Print hyperparameters
print(model_msg)
print(f'Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of workers: {NUM_WORKERS}')
print(f'Initial learning rate: {INITIAL_LR}')
print(f'Weight decay: {WEIGHT_DECAY}')
print(f'Loss function: {LOSS_FN}')
print(f'Scheduler: {SCHEDULER}\n')

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

# Assign class weights
if IMAGE_SIZE[0] == 256:
    # Based on resized imgs (256x256) - log scale of inverse class freqs
    room_weights = torch.tensor([0.0131, 0.0785, 0.0662, 0.0791, 0.0652, 0.0683, 0.0973, 0.0885, 0.1452, 0.1113, 0.1259, 0.0614], device=device)
    icon_weights = torch.tensor([0.0008, 0.0734, 0.0861, 0.0689, 0.0815, 0.1048, 0.0963, 0.0868, 0.1183, 0.1256, 0.1576], device=device)
else:
    # Based on resized imgs (512x512) - log scale of inverse class freqs
    room_weights = torch.tensor([0.0128, 0.0785, 0.0664, 0.0792, 0.0654, 0.0684, 0.0974, 0.0885, 0.1450, 0.1112, 0.1255, 0.0616], device=device)
    icon_weights = torch.tensor([0.0007, 0.0735, 0.0861, 0.0690, 0.0815, 0.1049, 0.0964, 0.0868, 0.1181, 0.1258, 0.1572], device=device)

# Initialize loss functions
if LOSS_FN == 'dice':
    room_criterion = DiceLoss().to(device)
    icon_criterion = DiceLoss().to(device)
elif LOSS_FN == 'weighted_dice':
    room_criterion = DiceLoss(weights=room_weights).to(device)
    icon_criterion = DiceLoss(weights=icon_weights).to(device)
elif LOSS_FN == 'combo':
    room_criterion = ComboLoss(weights=room_weights, alpha=0.5).to(device)
    icon_criterion = ComboLoss(weights=icon_weights, alpha=0.1).to(device)
else:
    raise ValueError('Invalid loss function')

heatmap_criterion = nn.MSELoss()
multitask_criterion = MultiTaskUncertaintyLoss().to(device)

# Weight decay to prevent overfitting
optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# Get from CLI arg
if SCHEDULER == 'onecycle':
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=INITIAL_LR * 10, epochs=EPOCHS, steps_per_epoch=len(train_loader))
elif SCHEDULER == 'cawr':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
else:
    raise ValueError('Invalid learning rate scheduler')

# To speed up training
scaler = torch.amp.GradScaler()

# Load best model checkpoint if available or train from scratch
checkpoint_path = f'saved_models/deeplab_{model.backbone_name}_{model.attention}.pt'

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

# Setup logging
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
FILE_NAME = f'deeplab_{model.backbone_name}_{model.attention}_{timestamp}'

os.makedirs('logs', exist_ok=True)
log_filepath = os.path.join('logs', f'{FILE_NAME}.log')
logging.basicConfig(filename=log_filepath, level=logging.INFO,format='%(asctime)s - %(message)s')

# Log hyperparameters
hyperparams_msg = (
    f'\n{model_msg}'
    f'\nImage size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}'
    f'\nBatch size: {BATCH_SIZE}'
    f'\nNumber of workers: {NUM_WORKERS}'
    f'\nInitial learning rate: {INITIAL_LR}'
    f'\nWeight decay: {WEIGHT_DECAY}'
    f'\nLoss function: {LOSS_FN}'
    f'\nScheduler: {SCHEDULER}\n'
)
logging.info(hyperparams_msg)

# TRAINING AND VALIDATION LOOP
train_loss_history = []
val_loss_history = []
room_mpa_history = []
room_miou_history = []
room_fwiou_history = []
icon_mpa_history = []
icon_miou_history = []
icon_fwiou_history = []

early_stopping_counter = 0
early_stopping_patience = 20

for epoch in range(start_epoch, EPOCHS):
    epoch_msg= f'EPOCH {epoch+1}/{EPOCHS}'
    print(epoch_msg)
    logging.info(epoch_msg)

    # Get train & val loss and metrics
    train_loss = train_epoch(model, train_loader, room_criterion, icon_criterion, 
                             heatmap_criterion, multitask_criterion, scaler, optimizer, scheduler, device=device)
    
    val_loss, val_room_metrics, val_icon_metrics = validate_epoch(model, val_loader, room_criterion, icon_criterion, 
                                                                  heatmap_criterion, multitask_criterion, room_metrics, icon_metrics, device)

    # Get actual values for each metric
    room_mpa = val_room_metrics['mpa'].item()
    room_miou = val_room_metrics['miou'].item()
    room_fwiou = val_room_metrics['fwiou'].item()

    icon_mpa = val_icon_metrics['mpa'].item()
    icon_miou = val_icon_metrics['miou'].item()
    icon_fwiou = val_icon_metrics['fwiou'].item()

    # For plotting
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
        f'\nTrain Loss: {train_loss:.4f}'
        f'\nVal Loss:   {val_loss:.4f}'
        f'\nRooms - mPA: {room_mpa:.4f}, mIoU: {room_miou:.4f}, fwIoU: {room_fwiou:.4f}'
        f'\nIcons - mPA: {icon_mpa:.4f}, mIoU: {icon_miou:.4f}, fwIoU: {icon_fwiou:.4f}'
        f'\nLearning Rate: {scheduler.get_last_lr()[0]}\n'
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

        save_msg = f'\nBest model saved with validation loss: {best_val_loss:.4f}\n'
        print(save_msg)
        logging.info(save_msg)
    else:
        early_stopping_counter += 1
        not_improved_msg = f'\nVal loss did not improve for {early_stopping_counter} epoch/s. Best loss is still: {best_val_loss:.4f}\n'
        print(not_improved_msg)
        logging.info(not_improved_msg)

    # Activate early stopping if no improvement threshold is reached
    if early_stopping_counter == early_stopping_patience:
        early_stop_msg = f'\nEarly stopping triggered after {epoch} epochs'
        print(early_stop_msg)
        logging.info(early_stop_msg)
        break

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')


# Visualize training results and save to PDF
os.makedirs('visualizations', exist_ok=True)
pdf_filepath = os.path.join('visualizations', f"{FILE_NAME}.pdf")
n = range(1, len(train_loss_history) + 1)

with PdfPages(pdf_filepath) as pdf:
    visualize(pdf, n,  train_loss_history, val_loss_history, 'Train Loss', 'Val Loss', 'Training vs Validation Loss', 'Epochs', 'Loss')
    visualize(pdf, n, room_mpa_history, icon_mpa_history, 'Room mPA', 'Icon mPA', 'Room vs Icon Mean Pixel Accuracy', 'Epochs', 'Mean Pixel Accuracy')
    visualize(pdf, n, room_miou_history, icon_miou_history, 'Room mIoU', 'Icon mIoU', 'Room vs Icon Mean Intersection over Union', 'Epochs', 'mIoU')
    visualize(pdf, n, room_fwiou_history, icon_fwiou_history, 'Room fwIoU', 'Icon fwIoU', 'Room vs Icon Frequency-Weighted IoU', 'Epochs', 'fwIoU')

print('Training completed successfully!')
