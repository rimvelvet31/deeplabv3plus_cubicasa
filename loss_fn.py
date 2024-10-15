import torch
import torch.nn as nn

class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()

        # Learnable parameter for each task
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))


    def forward(self, losses):
        """
        Forward pass to compute the total multi-task loss.
        Args:
            losses (list of tensors): List of individual task losses.
        Returns:
            total_loss (tensor): The final weighted multi-task loss.
        """

        # Tensor to avoid issues with autograd
        total_loss = torch.tensor(0.0, requires_grad=True).to(losses[0].device)

        for i, loss in enumerate(losses):
            precision_weight = torch.exp(-self.log_sigma[i])
            task_loss = precision_weight * loss + self.log_sigma[i]
            total_loss = total_loss + task_loss
        
        return total_loss