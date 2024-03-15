import os
import torch

gt = torch.randn((64,64))
gt = gt.unsqueeze(dim = -1)
print(gt.shape)