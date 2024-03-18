import os
import numpy as np

base_dir = r'D:\medical\brain\data\BraTS2020_Segmentation_multisegment\train\anomal/gt_npy_64_class12'
files = os.listdir(base_dir)
for file in files :
    file_dir = os.path.join(base_dir, file)
    arr = np.load(file_dir)
    print(f'arr.shape = {arr.shape} | np.unique = {np.unique(arr)}')