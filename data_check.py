import os
import numpy as np
from PIL import Image

base_dir = r'D:\medical\leader_polyp\bkai-igh-neopolyp/bkai-igh-neopolyp_sy'
phases = os.listdir(base_dir)
for phase in phases:
    phase_dir = os.path.join(base_dir, f'{phase}/anomal/mask_256')
    save_dir = os.path.join(base_dir, f'{phase}/anomal/mask_256_2')
    os.makedirs(save_dir, exist_ok=True)

    files = os.listdir(phase_dir)
    for file in files:
        name = file.split('.')[0]
        org_path = os.path.join(phase_dir, file)
        arr = np.load(org_path)
        print(arr.shape)
        #re_path = os.path.join(save_dir, f'{name}.npy')
        #os.rename(org_path, re_path)
