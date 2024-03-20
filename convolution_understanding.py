from torch import nn
import torch
input = torch.randn(1,1,4,4)
sample_conv = nn.Conv2d(1,2,kernel_size = 3, padding=(1,1))
output = sample_conv(input) # 1,2,4,4
print(output.shape)