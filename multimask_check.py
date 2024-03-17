import os
import numpy as np
import nibabel as nib
import glob
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# [1] scaler to change image pixel range
# [2]
# [3] slicing on slice axis
# [4] combine and reshape to remove bad point (240 -> 128)

def scaling_img(raw_img, scaler):
    raw = raw_img.reshape(-1, raw_img.shape[-1])  # [all pixel values, angles]
    raw = scaler.fit_transform(raw)  # set value from 0 ~ 1
    raw_img = raw.reshape(raw_img.shape)  # 240,240,155
    return raw_img
"""
def main():

    print(f'\n step0. save dir')
    dataset_path = r'D:/medical/brain/data/BraTS2020_Segmentation_multisegment'
    phases = os.listdir(dataset_path)
    for phase in phases :
        if phase == 'train' :
            phase_dir = os.path.join(dataset_path, phase)
            folders = os.listdir(phase_dir)
            for folder in folders :
                folder_dir = os.path.join(phase_dir, folder)
                npy_folder = os.path.join(folder_dir, 'gt_npy_64')
                gt_save_folder = os.path.join(folder_dir, 'gt_rgb_64')
                os.makedirs(gt_save_folder, exist_ok = True)
                files = os.listdir(npy_folder)
                for file in files :
                    name = os.path.splitext(file)[0]
                    file_dir = os.path.join(npy_folder, file)
                    arr = np.load(file_dir)
                    h,w = arr.shape
                    arr_cat = to_categorical(arr)
                    mask = np.zeros((h, w, 3))
                    if arr_cat.shape[-1] != 1 :
                        arr_cat = arr_cat[:,:,1:]
                        arr_cat = np.where(arr_cat == 0, 0, 255)
                        c_ = arr_cat.shape[-1]
                        mask[:,:,:c_] = arr_cat
                    mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
                    mask.save(os.path.join(gt_save_folder, f'{name}.jpg'))

"""
def main() :

    a = np.zeros((64,64, 4))





if __name__ == '__main__' :
    main()