import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class MulticlassLoss(nn.Module):

    '''
    Multi-class Focal loss implementation
    '''

    def __init__(self, gamma=2, focal_loss = True):
        super(MulticlassLoss, self).__init__()
        self.gamma = gamma
        self.focal_loss = focal_loss

    def forward(self, input, target):
        """
        input: [N, C] -> raw probability
        target: [N, ] -> 0 , ... , C-1 --> C class index
        """
        # [1] calculate class by weight
        # input is probability
        logpt = F.log_softmax(input, dim=1)
        if self.focal_loss:
            logpt = torch.exp(logpt) # -> probability
            logpt = ((1 - logpt) ** self.gamma) * logpt # probability refered

        # [2] what is nll_loss
        loss = F.nll_loss(logpt,
                          target.type(torch.LongTensor).to(logpt.device),)
        return loss

def dice_coeff(input: Tensor,
               target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    input = input.flatten(start_dim=0, end_dim=1)
    target = target.flatten(start_dim=0, end_dim=1)
    return dice_coeff(input, target, reduce_batch_first,  # False
                      epsilon)


def dice_loss(input: Tensor,
              target: Tensor,
              multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # good model have high dice_coefficient
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    dice_coefficient = fn(input, target, reduce_batch_first=True)
    dice_loss = 1 - dice_coefficient
    return dice_loss

