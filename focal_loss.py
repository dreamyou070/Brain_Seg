import torch.nn.functional as F
import torch

input = torch.randn((64,4))
# [1]
logpt = F.log_softmax(input, dim=1)
pt = torch.exp(logpt)
print(f'pt = {pt}')
# [2]
pt2 = F.softmax(input, dim=1)
print(f'pt2 = {pt2}')