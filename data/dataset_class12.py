import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2
from tensorflow.keras.utils import to_categorical

anomal_p = 0.03

def passing_mvtec_argument(args):
    global argument

    argument = args

class TestDataset_Multi(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240,240),
                 tokenizer=None,
                 caption : str = "necrotic, edema, tumor",
                 latent_res : int = 64,
                 num_classes = 4) :

        CLASSES = ['non', 'necrotic', 'edema', 'enhancing tumor']
        PALETTE = torch.tensor([ (0, 0, 0),
                                 (244, 243, 131),
                                 (137, 28, 157),
                                 (150, 255, 255),])

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        folders = os.listdir(self.root_dir)
        for folder in folders :
            if folder == 'anomal' :
                folder_dir = os.path.join(self.root_dir, folder)
                rgb_folder = os.path.join(folder_dir, 'xray')
                gt_folder = os.path.join(folder_dir, 'gt_npy_64')
                images = os.listdir(rgb_folder)
                for image in images :
                    name, ext = os.path.splitext(image)
                    image_paths.append(os.path.join(rgb_folder, image))
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))

        self.resize_shape = resize_shape
        self.tokenizer = tokenizer
        self.caption = caption
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.latent_res = latent_res
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)
        return pil

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        # [1] base
        img_path = self.image_paths[idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1],type='RGB')  # np.array,

        # [2] gt dir
        gt_path = self.gt_paths[idx]
        gt_array = np.load(gt_path)  # (240,240)
        gt_vector = torch.from_numpy(gt_array).flatten()

        # (1) 64
        base_gt = np.zeros((self.latent_res , self.latent_res, self.num_classes))
        gt = to_categorical(gt_array)
        h, w, c = gt.shape
        base_gt[:, :, :c] = gt # [64,64,c]
        base_gt = torch.from_numpy(base_gt)
        base_gt = base_gt.permute(2,0,1) # [c,64,64]

        # [3] caption
        input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]

        return {'image': self.transform(img), # [3,512,512]
                "gt" : base_gt,                    # [64*64, 4]
                "gt_vector" : gt_vector,
                "input_ids" : input_ids}

class TrainDataset_Multi(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240,240),
                 tokenizer=None,
                 caption : str = "necrotic, edema, tumor",
                 latent_res : int = 64,
                 num_classes = 4) :

        CLASSES = ['non', 'necrotic', 'edema', 'enhancing tumor']
        PALETTE = torch.tensor([ (0, 0, 0),
                                 (244, 243, 131),
                                 (137, 28, 157),
                                 (150, 255, 255),])

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths = [], []
        self.feature_64_list, self.feature_32_list, self.feature_16_list = [], [] , []
        folders = os.listdir(self.root_dir)
        for folder in folders :
            if folder == 'anomal' :
                folder_dir = os.path.join(self.root_dir, folder)
                rgb_folder = os.path.join(folder_dir, 'xray')
                gt_folder = os.path.join(folder_dir, 'gt_npy_64_class12')
                feature_64_folder = os.path.join(folder_dir, 'feature_64')
                feature_32_folder = os.path.join(folder_dir, 'feature_32')
                feature_16_folder = os.path.join(folder_dir, 'feature_16')
                images = os.listdir(rgb_folder)
                for image in images :
                    name, ext = os.path.splitext(image)
                    image_paths.append(os.path.join(rgb_folder, image))
                    gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))
                    self.feature_64_list.append(os.path.join(feature_64_folder, f'{name}.pt'))
                    self.feature_32_list.append(os.path.join(feature_32_folder, f'{name}.pt'))
                    self.feature_16_list.append(os.path.join(feature_16_folder, f'{name}.pt'))


        self.resize_shape = resize_shape
        self.tokenizer = tokenizer
        self.caption = caption
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.latent_res = latent_res
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)
        return pil

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        # [1] base
        img_path = self.image_paths[idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,

        # [2] gt dir
        gt_path = self.gt_paths[idx]
        gt_array = np.load(gt_path)                      # [64,64]]
        gt_array = np.array(Image.fromarray(gt_array).resize((self.latent_res, self.latent_res)))
        gt_vector = torch.from_numpy(gt_array).flatten() # [4096]


        # (1) 64
        base_gt = np.zeros((self.latent_res*self.latent_res, self.num_classes))
        gt = to_categorical(gt_array.flatten()) # [4096,3]
        p, c = gt.shape # max = 3
        base_gt[:, :c] = gt # [64,64,c]
        base_gt = torch.from_numpy(base_gt)
        #base_gt = base_gt.permute(2,0,1) # [c,64,64]

        # featire_64
        feature_64 = self.feature_64_list[idx]
        feature_64 = torch.load(feature_64) # res,res,dim
        feature_64 = feature_64.permute(2,0,1)
        feature_32 = self.feature_32_list[idx]
        feature_32 = torch.load(feature_32).permute(2,0,1)
        feature_16 = self.feature_16_list[idx]
        feature_16 = torch.load(feature_16).permute(2,0,1)


        # [3] caption
        input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]

        return {'image': self.transform(img), # [3,512,512]
                "gt" : base_gt,               # [64*64, 3]
                "gt_vector" : gt_vector,      # [4096]
                "input_ids" : input_ids,
                "feature_64" : feature_64,
                "feature_32": feature_32,
                "feature_16": feature_16,
                }
#import torch
#base_dir = r'D:\medical\brain\data\BraTS2020_Segmentation_multisegment\train\anomal/sample094_42.pt'
#feature = torch.load(base_dir)
#print(feature.shape)