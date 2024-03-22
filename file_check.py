import os
import numpy as np
from PIL import Image
"""
base_filder = r'D:\medical\leader_polyp\bkai-igh-neopolyp/train'
image_folder = os.path.join(base_filder, 'image')
mask_folder = os.path.join(base_filder, 'mask')
mask_files = os.listdir(mask_folder)

image_list = os.listdir(mask_folder)

save_folder = r'D:\medical\leader_polyp\bkai-igh-neopolyp/train_saving'
os.makedirs(save_folder, exist_ok=True)
save_image_folder = os.path.join(save_folder, 'image_256')
os.makedirs(save_image_folder, exist_ok=True)
save_mask_folder = os.path.join(save_folder, 'mask_256')
os.makedirs(save_mask_folder, exist_ok=True)
for i, image in enumerate(mask_files):

    # [1]
    img = Image.open(os.path.join(image_folder, image)).resize((256,256))
    img.save(os.path.join(save_image_folder, f'sample_{i}.png'))

    # [2] mask
    mask_pil = Image.open(os.path.join(mask_folder, image)).resize((256,256))
    #mask_np = np.array(mask_pil)
    #base_np = np.zeros((256,256))
    #r_channel = mask_np[:,:,0]
    #g_channel = mask_np[:,:,1]
    #r_position = np.where(r_channel > 255/2, 1, 0)
    #g_position = np.where(g_channel > 255/2, 1, 0)
    #base_np = r_position + g_position * 2
    #print(f'{image} = {np.unique(base_np)}')
    mask_pil.save(os.path.join(save_mask_folder, f'sample_{i}.png'))
"""
base_filder = r'D:\medical\leader_polyp\bkai-igh-neopolyp/train_saving'
image_folder = os.path.join(base_filder, 'image_256')
mask_folder = os.path.join(base_filder, 'mask_256')
image_list = os.listdir(image_folder)
mask_list = os.listdir(mask_folder)
for mask_file in mask_list :
    mask_path = os.path.join(mask_folder, mask_file)
    mask_pil = Image.open(mask_path)
    mask_np = np.array(mask_pil)
    base_np = np.zeros((256,256))

    r_channel = mask_np[:,:,0]
    g_channel = mask_np[:,:,1]
    r_position = np.where(r_channel > 255/2, 1, 0)
    g_position = np.where(g_channel > 255/2, 1, 0)
    base_np = r_position + g_position * 2
    np.save(os.path.join(mask_folder, f'{mask_file}.npy'), base_np)
