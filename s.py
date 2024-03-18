import torch
import numpy as np
"""
masks_pred = torch.randn(4,64,64).cpu().numpy()
masks_pred = np.argmax(masks_pred, axis=0) # 64,64
position = np.where(masks_pred == 0, 1, 0)
position = np.expand_dims(position, axis=2) # 64,64,1
position = np.repeat(position, 3, axis=2)
position_color = position * [0,0,0]
print(position_color)
"""
a = np.zeros((64,64,3))
masks_pred = torch.randn((1,4,64,64))
import torch.nn.functional as F
masks_pred = F.softmax(masks_pred, dim=1).squeeze(0).detach().cpu().numpy() # 4,64,64

masks_pred = np.argmax(masks_pred, axis=0) #
print(f'masks_pred (64,64) = {masks_pred.shape}')