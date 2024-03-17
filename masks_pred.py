import torch

masks_pred = torch.randn(1,4,64,64)
masks_pred = masks_pred.permute(0,2,3,1) # 1,64,64,4
c = masks_pred.shape[-1]
masks_pred = masks_pred.view(-1,c)
print(masks_pred.shape)
