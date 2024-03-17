
import torch
from torch import nn
attn_score = torch.randn(64*64,4)
gt_vector = (torch.randn(64*64) * 0).type(torch.LongTensor)
crossentropy_loss_fn = nn.CrossEntropyLoss()
loss = crossentropy_loss_fn(attn_score, gt_vector)
print(loss)