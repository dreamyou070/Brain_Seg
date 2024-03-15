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

class TestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

class TrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 resize_shape=(240,240),
                 tokenizer=None,
                 caption : str = "necrotic, edema, tumor",
                 latent_res : int = 64) :

        """
            num_classes: 4
            """
        CLASSES = ['non', 'necrotic', 'edema', 'enhancing tumor']
        PALETTE = torch.tensor([ (0, 0, 0),
                                 (244, 243, 131),
                                 (137, 28, 157),
                                 (150, 255, 255),])

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths, object_masks = [], [],[]
        folders = os.listdir(self.root_dir)
        for folder in folders :
            folder_dir = os.path.join(self.root_dir, folder)
            rgb_folder = os.path.join(folder_dir, 'xray')
            gt_folder = os.path.join(folder_dir, 'gt')
            images = os.listdir(rgb_folder)
            for image in images :
                name, ext = os.path.splitext(image)
                image_path = os.path.join(rgb_folder, image)
                image_paths.append(image_path)
                gt_paths.append(os.path.join(gt_folder, f'{name}.npy'))

        self.resize_shape = resize_shape
        self.tokenizer = tokenizer
        self.caption = caption
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],
                                                                  [0.5]),])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.object_masks = object_masks
        self.latent_res = latent_res
        categories = self.CLASSES

    def __len__(self):
        
        return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)

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
        gt_path = self.gt_paths[idx] # 128,128,1
        gt_array = np.load(gt_path)  # categorise (128,128,1) -> (128,128,4)

        # (1) 64
        gt_64 = np.array(Image.fromarray(gt_array.astype(np.uint8)).resize((64, 64))).astype(np.uint8)
        gt_64 = torch.from_numpy(to_categorical(gt_64))  # [240,240,4]

        # (2) 32
        gt_32 = np.array(Image.fromarray(gt_array.astype(np.uint8)).resize((32, 32))).astype(np.uint8)
        gt_32 = torch.from_numpy(to_categorical(gt_32) ) # [240,240,4]

        # (3) 32
        gt_16 = np.array(Image.fromarray(gt_array.astype(np.uint8)).resize((16, 16))).astype(np.uint8)
        gt_16 = torch.from_numpy(to_categorical(gt_16) ) # [240,240,4]

        # (4) 32
        gt_8 = np.array(Image.fromarray(gt_array.astype(np.uint8)).resize((8, 8))).astype(np.uint8)
        gt_8 = torch.from_numpy(to_categorical(gt_8))  # [240,240,4]

        # [3] caption
        input_ids, attention_mask = self.get_input_ids(self.caption)  # input_ids = [77]

        return {'image': self.transform(img),               # [3,512,512]
                "gt_64" : gt_64,
                "gt_32": gt_32,
                "gt_16": gt_16,
                "gt_8": gt_8,
                "input_ids" : input_ids}