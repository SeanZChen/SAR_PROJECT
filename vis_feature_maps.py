# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:25:39 2020

@author: czx
"""

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize, Resize, CenterCrop, RandomRotation, RandomCrop

def main():
    from modeling.FullyConv import AConv
    net = AConv()
    root = 'checkpoints\\Fully_Conv_Rotate20_2deg\\best_checkpoint'
    cpt = torch.load(root)
    net.load_state_dict(cpt)
    net.eval()
    
    trans = Compose(
            [
                CenterCrop(size=(88, 88)),
                # RandomResizedCrop(inp_size, scale=inp_scale),
                # RandomHorizontalFlip(),
                ToTensor(),
                # Normalize(mean=MEAN, std=STD)
            ]
        )
    
    img_root = 'data\\TRAIN\\BTR_60\\HB03792.jpg'
    img = Image.open(img_root).convert('RGB')
    img = trans(img)
    img = img.unsqueeze(0)
    x = net.bn1(net.conv1(img))
    
    x1 = x[:,0,:,:]
    x1 = x1.squeeze(0)
    out_img = transforms.ToPILImage()(x1)
    out_img.show()
    
    
        
if __name__ == "__main__":
    main()