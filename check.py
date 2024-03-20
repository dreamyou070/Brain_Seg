import torch
from torch.nn import functional as F
import numpy as np
masks_pred = torch.randn(1,4,128,128)
masks_pred = F.softmax(masks_pred, dim=1).squeeze(0).detach().cpu().numpy()  # 4,128,128
masks_pred = np.argmax(masks_pred, axis=0) # [128,128], unique = 0,1,2,3
position = np.expand_dims(masks_pred, axis=2)  # 128,128,1
position = np.repeat(position, 3, axis=2)    # 128,128,3
color = [1,1,1]
color_map = position * color
print(color_map)