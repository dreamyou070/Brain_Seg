import os
from scipy import io
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
def resize(item,size):
    T = torchvision.transforms.Resize(size=(size,size),
                                      interpolation=transforms.InterpolationMode.BILINEAR,
                                      antialias=True)
    return T(item)
def check_for_nonzero(item):
    if torch.count_nonzero(item)!=0:
        return True
    return False



#base_dir = r'D:\medical\oct\2015_BOE_Chiu\2015_BOE_Chiu'
base_dir = r'D:\medical\oct\2015_BOE_Chiu\sub'
files = os.listdir(base_dir)
size = 512
images, masks = torch.tensor([]),torch.tensor([])
for file in files:
    file_path = os.path.join(base_dir, file)
    mat = io.loadmat(file_path)

    # [1] image
    # array, 0 ~ 255, shape = 496,768,61
    # change dim to # 61, 496, 768
    # normalize to 0 ~ 1
    # bachwise
    img_arr = np.transpose(mat['images'], (2, 0, 1)) / 255
    images = np.expand_dims(img_arr, 0) # 1,61,496,768

    # [2] masks
    # original shape =496,768,61 (all nan value)
    mask = mat['manualFluid1']
    y = np.transpose(mask, (2, 0, 1))
    masks = np.expand_dims(np.nan_to_num(y), 0) # non be 0, shape = [1,61,496,768]

    # convert to tensor
    images = torch.as_tensor(images, dtype=torch.float32)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    images = resize(images, size)
    masks = resize(masks, size)

    # saving (total 61 axis)
    for idx in range(images.shape[1]):
        mask = masks[0][idx]
        if check_for_nonzero(mask):
            temp1 = images[::, idx, ::]
            temp2 = masks[0, idx, ::].unsqueeze(0)
            # make image 3 channel instead of 1 -> (1, 3, H, W)
            img = torch.cat([temp1] * 3).unsqueeze(0)
            #                     print(img.shape)

            images = torch.cat((images, img))
            masks = torch.cat((masks, temp2))
