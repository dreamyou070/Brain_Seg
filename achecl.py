import os
import torch

attention_loss = [torch.randn(8,64), torch.randn(8,64)]
attn_loss = torch.stack(attention_loss, dim=0)#.mean(dim=0)
print(attn_loss.shape)