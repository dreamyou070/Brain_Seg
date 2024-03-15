import torch
attn_score = torch.randn(8, 64*64, 4)
normal_map = attn_score[:,:,0].squeeze()#.mean()
print(normal_map.shape)