import torch
import numpy as np
from torch import nn
input = torch.randn((10,2))
sigmoid = nn.Sigmoid()
binary_gt_flat = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
binary_gt_flat = torch.nn.functional.one_hot(binary_gt_flat.to(torch.int64), num_classes=2).to(input.dtype)
bce_loss = torch.nn.BCELoss()
print(binary_gt_flat.shape)
print(binary_gt_flat)
output = bce_loss(sigmoid(input),
                  binary_gt_flat)