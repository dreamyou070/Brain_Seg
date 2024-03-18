import torch.nn.functional as F
import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
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
    return dice_coeff(input,
                      target,
                      reduce_batch_first, # False
                      epsilon)


input = torch.randn(1,4,64,64)
target = torch.randn(1,4,64,64) * 0
input = input.flatten(start_dim=0, end_dim=1)
target = target.flatten(start_dim=0, end_dim=1)
reduce_batch_first= False
sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3) # (-1,-2)
inter = 2 * (input * target).sum(dim=sum_dim)                   # channel wise
sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)     # channel wise
sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
dice = (inter + 1e-6) / (sets_sum + 1e-6)
dice = dice.mean()
print(dice)