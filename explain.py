# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:25:31 2020

@author: czx
"""

import PIL
import os
import numpy as np
import cv2
import random

train_root = 'data_Rotate45_2degree/TRAIN'
test_root = 'data/TEST'
img_queue = []

def get_img(dir):
    if dir.endswith('.jpg'):
        img = cv2.imread(dir)
        h,w,c = img.shape
        center_w = w//2
        center_h = h//2
        '''
        if random.random()<0.5:
            img = cv2.flip(img, 0)
        '''
        img_queue.append(img[center_h-44:center_h+44,center_w-44:center_w+44])
        return
    else:
        for filename in os.listdir(dir):
            get_img(os.path.join(dir, filename))
        
if __name__ == '__main__':
    get_img(train_root)
    res = np.zeros((88,88,3))
    for img in img_queue:
        res += img/len(img_queue)
    cv2.imwrite('avg_Rotate45train.jpg', res)
