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


class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, is_training=True):
        super(ImageNetDataset, self).__init__()
        data_root = 'rgb_data'
        self.is_training = is_training
        img_dirs = []
        labels = []
        '''
        if is_training:
            self.data_root = os.path.join(data_root, 'train')
        else:
            self.data_root = os.path.join(data_root, 'val')
        '''
        if is_training:
            meta = open(os.path.join(data_root, 'train_rgb_list.txt'), 'r')
        else:
            meta = open(os.path.join(data_root, 'test_rgb_list.txt'), 'r')
        # folders = os.listdir(self.data_root)
        # folders.sort()
        lines = meta.readlines()
        for idx, line in enumerate(lines):
            s = line.split(' ')
            img_dirs.append(os.path.join(data_root, s[0]))
            labels.append(int(s[1]))
        self.img_dirs = img_dirs
        self.labels = labels
        self.transform = self.transforms(is_train=is_training)

    def transforms(self, inp_size=100, is_train=False):
        if is_train:
            return Compose(
                [
                    Resize(size = 128),
                    # RandomRotation(90),
                    # CenterCrop(size=98),
                    # RandomCrop(size=88),
                    CenterCrop(size=88),
                    # RandomCrop(size=inp_size),
                    # RandomResizedCrop(inp_size, scale=(0.5, 1)),
                    # RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ToTensor(),
                    # Normalize(mean=MEAN, std=STD)
                ]
            )
        return Compose(
            [
                Resize(size = 128),
                # RandomRotation(45),
                CenterCrop(size=88),
                # RandomResizedCrop(inp_size, scale=inp_scale),
                # RandomHorizontalFlip(),
                ToTensor(),
                # Normalize(mean=MEAN, std=STD)
            ]
        )

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):
        rgb_img = Image.open(self.img_dirs[index]).convert('RGB')
        # print('Loading From ImageNet')
        '''
        if self.train:
            rgb_img = self.addSaltNoise(rgb_img)
        '''
        
        label_id = self.labels[index]

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label_id