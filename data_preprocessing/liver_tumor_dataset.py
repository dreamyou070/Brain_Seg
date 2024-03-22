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

base_dir = r'D:\medical\brain\data\ISLES-2022\ISLES-2022\derivatives'
folders = os.listdir(base_dir)
for folder in folders :
    folder_dir = os.path.join(base_dir, f'{folder}/ses-0001')
    files = os.listdir(folder_dir)
    for file in files :
        if os.path.splitext(file)[-1] == '.gz' :
            file_dir = os.path.join(folder_dir, file)
            proxy = nib.load(file_dir)
            arr = proxy.get_fdata()  # [112,112,73], 0~1 (all different shape)
            print(arr.shape)




