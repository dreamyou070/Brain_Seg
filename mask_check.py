import os, shutil
import numpy as np

base_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128/train/anomal/'
test_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain/BraTS2020_Segmentation_128/test'
os.makedirs(test_folder, exist_ok = True)
test_folder = os.path.join(test_folder, 'anomal')
os.makedirs(test_folder, exist_ok = True)
test_img_folder = os.path.join(test_folder, 'image_128')
os.makedirs(test_img_folder, exist_ok = True)
test_mask_folder = os.path.join(test_folder, 'mask_128')
os.makedirs(test_mask_folder, exist_ok = True)

image_folder = os.path.join(base_folder, 'image_128')
mask_folder = os.path.join(base_folder, 'mask_128')
masks = os.listdir(mask_folder)
images = os.listdir(image_folder)
test_num = int(len(images)*0.2)
for i, file in enumerate(images) :
    name, ext = os.path.splitext(file)
    org_img_dir = os.path.join(image_folder, file)
    org_npy_dir = os.path.join(mask_folder, f'{name}.npy')
    if i < test_num :
        new_img_dir = os.path.join(test_img_folder, file)
        new_npy_dir =os.path.join(test_mask_folder, f'{name}.npy')