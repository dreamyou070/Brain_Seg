import os
import numpy as np
from PIL import Image
import torch

binary_gt_flat = torch.Tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).long()
binary_gt_flat = torch.nn.functional.one_hot(binary_gt_flat.to(torch.int64), num_classes=2)
print(binary_gt_flat.shape)