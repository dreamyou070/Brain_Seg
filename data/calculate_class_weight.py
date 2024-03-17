import os
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import class_weight
base_folder = r'D:/medical/brain/data/BraTS2020_Segmentation_multisegment/train/anomal/gt_npy_64'
files = os.listdir(base_folder)
train_masks = []

for i, file in enumerate(files) :
    file_dir = os.path.join(base_folder, file)
    gt_arr = np.load(file_dir)
    train_masks.append(gt_arr)

train_masks = np.array(train_masks) # num, H, W
n,h,w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1)

train_masks_input = np.expand_dims(train_masks, axis=3)
train_masks_cat = to_categorical(train_masks_input) # Num, 240,240,4
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes = np.unique(train_masks_reshaped),
                                                  y = train_masks_reshaped)
np.save('class_weights.npy',class_weights)