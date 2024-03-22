import os
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage import io
from PIL import Image
def scaling_img(raw_img, scaler):
    raw = raw_img.reshape(-1, raw_img.shape[-1])  # [all pixel values, angles]
    raw = scaler.fit_transform(raw)  # set value from 0 ~ 1
    raw_img = raw.reshape(raw_img.shape)  # 240,240,155
    return raw_img

scaler = MinMaxScaler()

base_dir = r'D:\medical\acdc\database'
#training_dir = os.path.join(base_dir, 'training')
testing_dir = os.path.join(base_dir, 'testing')
saving_training_dir = os.path.join(base_dir, 'testing_save')
os.makedirs(saving_training_dir, exist_ok = True)

patient_folders = os.listdir(testing_dir)
lengths = set()
for patient_folder in patient_folders :
    # [1] get image and mask
    patient_img_folder = os.path.join(testing_dir, f'{patient_folder}')
    image_save_folder = os.path.join(saving_training_dir, 'image_256')
    mask_save_folder = os.path.join(saving_training_dir, 'mask_256')
    os.makedirs(image_save_folder, exist_ok = True)
    os.makedirs(mask_save_folder, exist_ok = True)
    # [2] get image
    files = os.listdir(patient_img_folder)
    file_dict = {}
    for file in files :
        if '4d' in file :
            img_file = os.path.join(patient_img_folder, file)
            file_dict['img_file'] = img_file
        elif 'frame01' in file and 'gt' not in file :
            frame01_file = os.path.join(patient_img_folder, file)
            file_dict['frame01_file'] = frame01_file
        elif 'gt' in file and 'frame01' in file :
            frame01_gt_file = os.path.join(patient_img_folder, file)
            file_dict['frame01_gt_file'] = frame01_gt_file
        else :
            if 'frame' in file and 'gt' not in file :
                frame_else_file = os.path.join(patient_img_folder, file)
                file_dict['frame_else_file'] = frame_else_file
            if 'frame' in file and 'gt' in file:
                frame_else_gt_file = os.path.join(patient_img_folder, file)
                file_dict['frame_else_gt_file'] = frame_else_gt_file
    img_file = file_dict['img_file']
    frame01_file = file_dict['frame01_file']
    frame01_gt_file = file_dict['frame01_gt_file']
    frame_else_file = file_dict['frame_else_file']
    frame_else_gt_file = file_dict['frame_else_gt_file']

    img_arr = nib.load(img_file).get_fdata()
    frame01_arr = nib.load(frame01_file).get_fdata()
    frame01_gt_arr = nib.load(frame01_gt_file).get_fdata()
    frame_else_arr = nib.load(frame_else_file).get_fdata()
    frame_else_gt_arr = nib.load(frame_else_gt_file).get_fdata()
    #print(img_arr.shape, frame01_arr.shape, frame01_gt_arr.shape, frame_else_arr.shape, frame_else_gt_arr.shape)
    #print(img_arr.max(), frame01_arr.max(), frame01_gt_arr.max(), frame_else_arr.max(), frame_else_gt_arr.max())
    #print(np.unique(frame01_gt_arr), np.unique(frame_else_gt_arr))
    #break
    layers = frame01_arr.shape[-1]
    for layer in range(layers) :
        frame01_layer = frame01_arr[:,:,layer]
        frame01_gt_layer = frame01_gt_arr[:,:,layer]
        frame_else_layer = frame_else_arr[:,:,layer]
        frame_else_gt_layer = frame_else_gt_arr[:,:,layer]
        h,w = frame01_layer.shape
        base_length = 256
        if h < base_length :
            # padding
            pad = base_length - h
            pad = pad // 2
            frame01_layer = np.pad(frame01_layer, ((pad,pad),(0,0)))
            frame01_gt_layer = np.pad(frame01_gt_layer, ((pad,pad),(0,0)))
            frame_else_layer = np.pad(frame_else_layer, ((pad,pad),(0,0)))
            frame_else_gt_layer = np.pad(frame_else_gt_layer, ((pad,pad),(0,0)))
        if w < base_length :
            # padding
            pad = base_length - w
            pad = pad // 2
            frame01_layer = np.pad(frame01_layer, ((0,0),(pad,pad)))
            frame01_gt_layer = np.pad(frame01_gt_layer, ((0,0),(pad,pad)))
            frame_else_layer = np.pad(frame_else_layer, ((0,0),(pad,pad)))
            frame_else_gt_layer = np.pad(frame_else_gt_layer, ((0,0),(pad,pad)))
        # center crop
        h, w = frame01_layer.shape
        frame01_layer = frame01_layer[h//2-base_length//2:h//2+base_length//2,w//2-base_length//2:w//2+base_length//2]
        frame01_gt_layer = frame01_gt_layer[h//2-base_length//2:h//2+base_length//2,w//2-base_length//2:w//2+base_length//2]
        frame_else_layer = frame_else_layer[h//2-base_length//2:h//2+base_length//2,w//2-base_length//2:w//2+base_length//2]
        frame_else_gt_layer = frame_else_gt_layer[h//2-base_length//2:h//2+base_length//2,w//2-base_length//2:w//2+base_length//2]
        # scaling
        frame01_layer = scaling_img(frame01_layer, scaler)
        frame_else_layer = scaling_img(frame_else_layer, scaler)
        frame01_pil = Image.fromarray((frame01_layer * 255).astype(np.uint8)).resize((256, 256))
        frame_else_pil = Image.fromarray((frame_else_layer * 255).astype(np.uint8)).resize((256, 256))

        # save
        frame01_pil.save(os.path.join(image_save_folder, f'{patient_folder}_frame01_{layer}.jpg'))
        frame_else_pil.save(os.path.join(image_save_folder, f'{patient_folder}_frame_else_{layer}.jpg'))

        frame01_gt_layer_dir = os.path.join(mask_save_folder, f'{patient_folder}_frame01_{layer}.npy')
        #print(frame01_gt_layer.shape)
        np.save(frame01_gt_layer_dir,frame01_gt_layer)
        frame_else_gt_layer_dir = os.path.join(mask_save_folder, f'{patient_folder}_frame_else_{layer}.npy')
        np.save(frame_else_gt_layer_dir,frame_else_gt_layer)

#lengths = list(lengths)
#print(max(lengths))
# 64 to 512
