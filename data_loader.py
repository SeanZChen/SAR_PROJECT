# ============================================
__author__ = "CHEN Zixuan"
__maintainer__ = "CHEN Zixuan"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize, Resize, CenterCrop, RandomRotation, RandomCrop
import numpy as np
import random

'''
This is a (untested) sample file for supporting custom datasets
'''

# let us assume that custom dataset has two classes, "Person" and "Background". Background is everything except persons
CUSTOM_DATASET_CLASS_LIST = ['background', 'person']


class SARDataset(torch.utils.data.Dataset):

    def __init__(self, root, inp_size=100, is_training=True):
        super(SARDataset, self).__init__()
        cats = ['2S1', 'BRDM_2', 'BTR_60', 'D7', 'BMP2', 'BTR70', 'T72', 'T62', 'ZIL131', 'ZSU_23_4']
        cats_dict = {cats[i]:i for i in range(len(cats))}
        # rgb_root_dir = os.path.join(root, 'images')
        self.train = is_training
        if self.train:
            img_root = os.path.join(root, 'TRAIN')
        else:
            img_root = os.path.join(root, 'TEST')

        self.images = []
        self.labels = []
        # with open(data_file, 'r') as lines:
        '''
            for line in lines:
                # line is a comma separated file that contains mapping between RGB image and class iD
                # <image1.jpg>, Class_ID
                line_split = line.split(',') # index 0 contains rgb location and index 1 contains label id
                rgb_img_loc = rgb_root_dir + os.sep + line_split[0].strip()
                label_id = int(line_split[1].strip()) #strip to remove spaces
                assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.labels.append(label_id)
        '''
        for folder in os.listdir(img_root):
            folder_dir = os.path.join(img_root, folder)
            for img_filename in os.listdir(folder_dir):
                if img_filename.endswith('jpg') or img_filename.endswith('JPG'):
                    img_file_path = os.path.join(folder_dir, img_filename)
                    self.images.append(img_file_path)
                    self.labels.append(cats_dict[folder])

        self.transform = self.transforms(inp_size=inp_size, is_train = is_training)

    def transforms(self, inp_size=100, is_train=False):
        if is_train:
            return Compose(
                [
                    # RandomRotation(180),
                    # CenterCrop(size=inp_size+20),
                    CenterCrop(size=inp_size),
                    # RandomCrop(size=inp_size),
                    # RandomResizedCrop(inp_size, scale=(0.5, 1)),
                    # RandomHorizontalFlip(p=0.5),
                    # RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    # Normalize(mean=MEAN, std=STD)
                ]
            )
        return Compose(
            [
                CenterCrop(size=(inp_size, inp_size)),
                # RandomResizedCrop(inp_size, scale=inp_scale),
                # RandomHorizontalFlip(),
                ToTensor(),
                # Normalize(mean=MEAN, std=STD)
            ]
        )
    
    def addSaltNoise(self, pil_img, p=.5):
        a = random.random()
        if a>p:
            return pil_img
        np_img = np.array(pil_img)
        noise_pos = random.randint(100, 1000)
        rows,cols,dims = np_img.shape
        for i in range(noise_pos):
            x=np.random.randint(0,rows)
            y=np.random.randint(0,cols)
            # t=random.randint(0,2)
            np_img[x,y,:]=255
            new_pil_img = Image.fromarray(np_img.astype('uint8')).convert('RGB')
        return new_pil_img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert('RGB')
        
        '''
        if self.train:
            rgb_img = self.addSaltNoise(rgb_img)
        '''
        
        label_id = self.labels[index]

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label_id