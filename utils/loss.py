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


def deactivating_loss(input, target, ignore_idx):
    # input shape = [N, C] -> raw probability
    # target shape = [N, ] -> 0 , ... , C-1 --> C class index

    # input = torch.randn(1,4,256,256)
    # input = torch.softmax(input, dim=1)
    # input = input.permute(0,2,3,1) # make from [1,4,3,3] to [1*3*3,4]
    # input = input.view(-1, input.shape[-1])
    # target = torch.randint(0,4,(9,))
    # ignore_idx = 0
    # penalty_loss = deactivating_loss(input, target, ignore_idx)
    # print(penalty_loss)

    input = torch.softmax(input, dim=1)
    input = input.permute(0, 2, 3, 1)   # make from [1,res,res,3]
    class_num = input.shape[-1]
    input = input.view(-1, class_num)
    penalty_loss_list = []
    for class_idx in range(class_num):
        if class_idx != ignore_idx:
            # [1] input probability
            target_input = input[:, class_idx]
            target_position = torch.where(target == class_idx, 1, 0)
            un_position = 1 - target_position
            total_input = torch.ones_like(target_input)
            un_position_score = target_input * un_position
            un_position_loss = (un_position_score / total_input) ** 2
            penalty_loss_list.append(un_position_loss)
    penalty_loss = torch.stack(penalty_loss_list).mean(dim=-1)  # 3, 9 -> class by averaging
    penalty_loss = torch.mean(penalty_loss)  # total averaging
    return penalty_loss