import os, shutil
import numpy as np
from torch import nn
import torch
from torch import nn

# dim = 4
batch_layer = nn.BatchNorm2d(4) # weight, bias
x = torch.randn(1,4,64,64)
importance  = batch_layer(x)
#print(output)
print(batch_layer.__dict__['_parameters'])