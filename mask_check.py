import os, shutil
import numpy as np

base_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128/train/anomal/'
image_folder = os.path.join(base_folder, 'image_128')
mask_folder = os.path.join(base_folder, 'mask_128')
masks = os.listdir(mask_folder)
print(f'present mask num = {len(masks)}')
images = os.listdir(image_folder)
for file in images :
    name, ext = os.path.splitext(file)
    if ext == '.npy' :
        org_img_dir = os.path.join(image_folder, file)
        os.remove(org_img_dir)
