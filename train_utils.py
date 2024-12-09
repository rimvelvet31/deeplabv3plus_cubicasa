import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


def process_batch(batch, device='cuda'):
    images = batch['image'].float().to(device, non_blocking=True)
    room_labels = batch['label'][:, 21].long().to(device, non_blocking=True)
    icon_labels = batch['label'][:, 22].long().to(device, non_blocking=True)
    heatmap_labels = batch['label'][:, 0:21].to(device, non_blocking=True)

    return images, room_labels, icon_labels, heatmap_labels


def train_epoch(model, 
                dataloader, 
                room_criterion, 
                icon_criterion, 
                heatmap_criterion, 
                multitask_criterion, 
                scaler, 
                optimizer, 
                device='cuda'):
    
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc='Training'):
        images, room_labels, icon_labels, heatmap_labels = process_batch(batch, device)

        # Avoid accumulating gradients that can lead to incorrect optimization
        optimizer.zero_grad()

        # Forward pass using mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            room_output, icon_output, heatmap_output = model(images)

            # Compute individual losses then feed into multi-task loss
            room_loss = room_criterion(room_output, room_labels)
            icon_loss = icon_criterion(icon_output, icon_labels)
            heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)

            # print(f'Room Loss: {room_loss.item()}, Icon Loss: {icon_loss.item()}, Heatmap Loss: {heatmap_loss.item()}')

            multitask_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

        # Backpropagation
        scaler.scale(multitask_loss).backward()

        # To prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        total_loss += multitask_loss.item()

    epoch_loss = total_loss / len(dataloader)
    return epoch_loss


def validate_epoch(model,
                   dataloader,
                   room_criterion,
                   icon_criterion,
                   heatmap_criterion,
                   multitask_criterion,
                   room_metrics,
                   icon_metrics,
                   device='cuda'):
    
    # Set model to evaluation mode to avoid updating weights
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images, room_labels, icon_labels, heatmap_labels = process_batch(batch, device)

            room_output, icon_output, heatmap_output = model(images)
            
            room_loss = room_criterion(room_output, room_labels)
            icon_loss = icon_criterion(icon_output, icon_labels)
            heatmap_loss = heatmap_criterion(heatmap_output, heatmap_labels)

            multitask_loss = multitask_criterion([room_loss, icon_loss, heatmap_loss])

            total_loss += multitask_loss.item()

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


def visualize(pdf, x_data, y1_data, y2_data, y1_legend, y2_legend, title, xlabel, ylabel, figsize=(10, 6), grid=True):
    plt.figure(figsize=figsize)
    plt.plot(x_data, y1_data, label=y1_legend)
    plt.plot(x_data, y2_data, label=y2_legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid)
    pdf.savefig()
    plt.close()
