"""
Focal Loss for handling severe class imbalance
Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
Used in medical imaging when some classes are rare
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Addresses class imbalance by down-weighting easy examples
    and focusing on hard, misclassified examples
    
    Args:
        alpha: Class weights (tensor of size [num_classes])
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits from model
            targets: (batch_size,) - class labels
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
