import os
import torch

attention_scores = torch.randn(8,64*64, 77)
attn = attention_scores.softmax(dim=-1)[:, :, :2]
print(attn.shape)