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