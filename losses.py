import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiceLoss(nn.Module):
    def __init__(self, weights=None, smooth=1e-7):
        """
        Dice Loss with optional class weights (transforms into Generalized Dice Loss).
        Args:
            weights (torch.Tensor or None): Weights for each class. If None, all classes are weighted equally.
            smooth (float): Smoothing term to avoid division by zero.
        """
        super().__init__()
        self.weights = weights
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Compute Dice Loss / Generalized Dice Loss if weights are provided.
        Args:
            predictions: Tensor of shape [B, C, H, W] containing predicted logits.
            targets: Tensor of shape [B, H, W] containing target class indices.
        Returns:
            dice_loss: Scalar Dice Loss value.
        """
        # Convert predictions to probabilities
        predictions = torch.softmax(predictions, dim=1)  # [B, C, H, W]

        # Flatten the predictions for computation
        B, C, H, W = predictions.size()
        predictions_flat = predictions.view(B, C, -1)  # [B, C, H*W]
        
        # One-hot encode the targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=C)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        targets_flat = targets_one_hot.view(B, C, -1)  # [B, C, H*W]

        # Compute intersection and union
        intersection = torch.sum(predictions_flat * targets_flat, dim=2)  # [B, C]
        union = torch.sum(predictions_flat, dim=2) + torch.sum(targets_flat, dim=2)  # [B, C]

        # Dice score per class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]

        if self.weights is not None:
            # For Generalized Dice Loss, apply weights to each class's dice score
            weighted_dice_score = dice_score * self.weights.unsqueeze(0)  # [B, C]
            dice_loss = 1 - weighted_dice_score.mean()  # Average across batch and classes
        else:
            # For standard Dice Loss, average over batches and classes
            dice_loss = 1 - torch.mean(dice_score)

        return dice_loss


class JaccardLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        IoU Loss for segmentation tasks.
        :param reduction: How to reduce the loss across batch ('mean', 'sum', or 'none').
        """

        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute IoU Loss.
        :param logits: Predicted logits (before softmax), shape: (batch_size, num_classes, height, width).
        :param targets: Ground truth labels, shape: (batch_size, height, width).
        :return: IoU Loss.
        """

        # To avoid division by zero
        EPSILON = 1e-6

        # Convert model raw outputs to probabilities
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)

        # One-hot encode to shape (batch_size, num_classes, height, width)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute IoU
        intersection = torch.sum(probs * one_hot_targets, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_targets, dim=(2, 3)) - intersection
        iou = (intersection + EPSILON) / (union + EPSILON)

        # Compute loss
        iou_loss = 1 - iou

        # Aggregate loss
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        elif self.reduction == 'none':
            return iou_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, epsilon=1e-6, reduction='mean'):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.gamma = gamma  # Focusing parameter
        self.epsilon = epsilon  # Prevent division by zero
        self.reduction = reduction  # 'mean', 'sum', or None

    def forward(self, y_pred, y_true):
        # Input validation
        assert y_pred.dim() == 4, "Input must be 4D tensor (B, C, H, W)"
        assert y_true.dim() == 3, "Target must be 3D tensor (B, H, W)"
        
        # Apply softmax to get class probabilities
        y_pred = torch.softmax(y_pred, dim=1)
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()
        
        # Compute True Positives, False Positives, and False Negatives
        TP = torch.sum(y_pred * y_true_one_hot, dim=(0, 2, 3))  # Sum across batch, height, and width for each class
        FP = torch.sum(y_pred * (1 - y_true_one_hot), dim=(0, 2, 3))
        FN = torch.sum((1 - y_pred) * y_true_one_hot, dim=(0, 2, 3))

        # Compute the Tversky index for all classes
        tversky_index = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)

        # Compute the Focal Tversky loss for all classes
        focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)

        # Reduction
        if self.reduction == 'mean':
            return torch.mean(focal_tversky_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_tversky_loss)
        else:
            return focal_tversky_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        """
        Focal Loss
        :param alpha: Tensor or scalar. Class weighting. Use a tensor of shape [num_classes] for class-specific weights.
        :param gamma: Focusing parameter (default: 2).
        :param reduction: Reduction method ('mean', 'sum', or 'none').
        """

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute Focal Loss.
        :param logits: Tensor of shape (batch_size, num_classes, height, width). Raw model outputs (logits).
        :param targets: Tensor of shape (batch_size, height, width). Ground truth labels (indices).
        :return: Focal loss value (scalar).
        """

        # Compute cross-entropy loss without reduction (for each pixel)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # Shape: (batch_size, height, width)
        
        # Compute log-softmax and probabilities
        log_probs = F.log_softmax(logits, dim=1)  # Shape: (batch_size, num_classes, height, width)
        targets_unsqueezed = targets.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        
        # Gather the log probabilities corresponding to the target class
        pt = torch.exp(log_probs.gather(1, targets_unsqueezed).squeeze(1))  # Shape: (batch_size, height, width)

        # Compute the focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.numel() > 1:  # Class-specific alpha
                alpha_t = self.alpha[targets]
            else:  # Scalar alpha
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-7, weights=None):
        """
        Combo Loss: Weighted combination of Cross-Entropy and Dice Loss.
        Args:
            alpha (float): Weight for balancing Cross-Entropy and Dice Loss. 
                           alpha=1 uses only CE, alpha=0 uses only Dice.
            smooth (float): Smoothing term for Dice Loss to avoid division by zero.
            ce_weights (torch.Tensor or None): Weights for Cross-Entropy loss for class imbalance.
            dice_class_weights (torch.Tensor or None): Weights for Dice Loss per class.
        """

        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        """
        Compute Combo Loss.
        Args:
            predictions: Tensor of shape [B, C, H, W] (logits).
            targets: Tensor of shape [B, H, W] (class indices).
        Returns:
            combo_loss: Scalar value of Combo Loss.
        """

        # Compute individual losses
        ce_loss = self.ce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)

        # Combine the losses
        combo_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss
        return combo_loss


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()

        # Learnable parameter for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        # self.log_vars = nn.Parameter(torch.full((num_tasks,), -2.0))

    def forward(self, losses):
        # Used tensor to avoid issues with autograd
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

        for i, loss in enumerate(losses):
            # For numerical stability
            clamped_log_vars = torch.clamp(self.log_vars[i], min=-10, max=10)

            precision_weight = torch.exp(-clamped_log_vars)
            task_loss = precision_weight * loss + clamped_log_vars
            total_loss = total_loss + task_loss
        
        return total_loss
    