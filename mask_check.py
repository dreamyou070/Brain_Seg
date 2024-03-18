import os
import numpy as np
import torch
mask = np.array([[1,2,3],
                  [0,1,2],
                 [3,3,1]])
c0_mask = np.where(mask==0, 1, 0)
c1_mask = np.where(mask==1, 1, 0)
c2_mask = np.where(mask==2, 1, 0)
c3_mask = np.where(mask==3, 1, 0)
head = 8
dim = 40
c0_mask = np.expand_dims(c0_mask, axis=0)
c0_mask = np.repeat(c0_mask, repeats = head, axis=0, )
c0_mask = np.expand_dims(c0_mask, axis=3)
c0_mask = np.repeat(c0_mask, repeats = dim, axis=0, )
c0_mask = torch.tensor(c0_mask)
print(c0_mask.shape)