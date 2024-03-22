import os
from PIL import Image
import numpy as np

base_dir = r'D:\medical\leader_polyp\bkai-igh-neopolyp\train_resize'
test_dir = r'D:\medical\leader_polyp\bkai-igh-neopolyp\test_resize'
os.makedirs(test_dir, exist_ok = True)

image_folder = os.path.join(base_dir, 'image_256')
mask_folder = os.path.join(base_dir, 'mask_256')
image_files = os.listdir(image_folder)

test_image_folder = os.path.join(test_dir, 'image_256')
test_mask_folder = os.path.join(test_dir, 'mask_256')
os.makedirs(test_image_folder, exist_ok = True)
os.makedirs(test_mask_folder, exist_ok = True)
for i, img_file in enumerate(image_files) :
    if i < 180 :
        org_img_path = os.path.join(image_folder, img_file)
        org_mask_path = os.path.join(mask_folder, img_file.replace('.jpg', '.npy'))
        save_img_path = os.path.join(test_image_folder, img_file)
        save_mask_path = os.path.join(test_mask_folder, img_file.replace('.jpg', '.npy'))
        os.rename(org_img_path, save_img_path)
        os.rename(org_mask_path, save_mask_path)


"""
for image_file in image_files :
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).resize((256,256))
    image.save(os.path.join(save_img_folder, image_file.replace('.jpeg', '.jpg')))

    mask_path = os.path.join(mask_folder, image_file)
    mask = Image.open(mask_path).resize((256,256))
    mask_arr = np.array(mask)
    r_channel = mask_arr[:,:,0]
    g_channel = mask_arr[:,:,1]
    b_channel = mask_arr[:,:,2]
    base = np.zeros((mask_arr.shape[0], mask_arr.shape[1]))
    class1 = np.where(r_channel < 255/2, 0, 1).astype(np.uint8)
    base += class1
    class2 = np.where(g_channel < 255 / 2, 0, 1).astype(np.uint8)
    np.save(os.path.join(save_mask_folder, image_file.replace('.jpeg', '.npy')), base)
"""
