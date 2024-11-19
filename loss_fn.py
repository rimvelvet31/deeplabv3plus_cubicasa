import torch
import torch.nn as nn

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
    