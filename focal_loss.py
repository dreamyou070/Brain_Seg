from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



multi_class_focal_loss = FocalLoss()
input = torch.randn(64*64,4)
target = (torch.randn(64*64) * 0).type(torch.LongTensor)
loss = multi_class_focal_loss(input, target)
print(loss)