import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_class_weights(loader, num_classes, label_index):
    """Calculate class weights based on the inverse frequency of each class."""
    class_counts = np.zeros(num_classes)
    
    # Count the number of occurrences of each class
    for batch in tqdm(loader, desc="Calculating class weights"):
        labels = batch['label'][:, label_index].numpy().flatten()
        for i in range(num_classes):
            class_counts[i] += np.sum(labels == i)
    
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= np.sum(class_weights)  # Normalize weights

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks)) # Learnable parameter for each task

    def forward(self, losses):
        # Used tensor to avoid issues with autograd
        total_loss = torch.tensor(0.0, requires_grad=True).to(losses[0].device)

        # Normalize losses so that they are in the same range (alternatively, can also use scaling)
        # normalized_losses = [loss / torch.max(loss) for loss in losses]

        for i, loss in enumerate(losses):
            # For numerical stability
            clamped_log_sigma = torch.clamp(self.log_sigma[i], min=-10, max=10)

            precision_weight = torch.exp(-clamped_log_sigma)
            task_loss = precision_weight * loss + clamped_log_sigma
            total_loss = total_loss + task_loss
        
        return total_loss
    