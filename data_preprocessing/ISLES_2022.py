import nibabel as nib
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import cv2
import imageio
from tqdm.notebook import tqdm
from PIL import Image
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage import io
def scaling_img(raw_img, scaler):
    raw = raw_img.reshape(-1, raw_img.shape[-1])  # [all pixel values, angles]
    raw = scaler.fit_transform(raw)  # set value from 0 ~ 1
    raw_img = raw.reshape(raw_img.shape)  # 240,240,155
    return raw_img
scaler = MinMaxScaler()

base_dir = r'D:\medical\brain\data\ISLES-2022'
image_dir = os.path.join(base_dir, 'image')
mask_dir = os.path.join(base_dir, 'mask')

patient_folders = os.listdir(image_dir)
for patient_folder in patient_folders :
    # [1] get image and mask
    patient_img_folder = os.path.join(image_dir, f'{patient_folder}/ses-0001')

    # [2] get image
    flair_file = os.path.join(patient_img_folder, f'anat/{patient_folder}_ses-0001_FLAIR.nii.gz')
    adc_file = os.path.join(patient_img_folder, f'dwi/{patient_folder}_ses-0001_adc.nii.gz')
    dwi_file = os.path.join(patient_img_folder, f'dwi/{patient_folder}_ses-0001_dwi.nii.gz')
    # [3] get mask
    patient_mask_folder = os.path.join(mask_dir, f'{patient_folder}/ses-0001')
    mask_file = os.path.join(patient_mask_folder, f'{patient_folder}_ses-0001_msk.nii.gz')

    nii_flair = nib.load(flair_file).get_fdata()
    nii_adc = nib.load(adc_file).get_fdata()
    nii_dwi = nib.load(dwi_file).get_fdata()
    nii_mask = nib.load(mask_file).get_fdata() #binary mask
    scaling_img(raw_img, scaler)
