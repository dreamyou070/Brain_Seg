import os
import numpy as np
base_folder = r'D:\medical\brain\data\Br35H_Brain_Tumor_Detection_128\train\mask_128'
files = os.listdir(base_folder)
for file in files :
    file_dir = os.path.join(base_folder, file)
    arr = np.load(file_dir)
    unique_value = np.unique(arr)
    print(unique_value)