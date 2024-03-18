import os
import numpy as np
from PIL import Image
data_dir = r'D:\medical\brain\data\BraTS2020_Segmentation_multisegment\train\anomal\gt_npy_64_class12/sample094_51.npy'
arr = np.load(data_dir)
arr = np.array(Image.fromarray(arr).resize((64,64)))
print(arr.shape)
print(np.unique(arr))