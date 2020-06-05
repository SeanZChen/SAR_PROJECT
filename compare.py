##################################################
#
# author: Zixuan Chen
# Date: 2020/03/30
# Description: 
# Meaure the similarity between training set and 
# validation set
#
##################################################

import os
import cv2
import numpy as np
import argparse
import random
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt 

cats = ['2S1', 'BRDM_2', 'BTR_60', 'D7', 'BMP2', 'BTR70', 'T72', 'T62', 'ZIL131', 'ZSU_23_4']

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float")-imageB.astype("float"))**2)
    err /= imageA.shape[0]*imageA.shape[1]
    return err

def classify_gray_hist(image1,image2): 

    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
    # 可以比较下直方图 
    # plt.plot(range(256),hist1,'r') 
    # plt.plot(range(256),hist2,'b') 
    # plt.show() 
    # 计算直方图的重合度 
    degree = 0
    for i in range(len(hist1)): 
        if hist1[i] != hist2[i]: 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
        else: 
            degree = degree + 1
            degree = degree/len(hist1) 
    return degree[0]

def calc(root1, root2):
    print(root1, root2)
    list1 = []
    list2 = []
    for root, dir, file_dir in os.walk(root1):
        # if file_dir.endswith('.jpg'):
        for img_file in file_dir:
            if 'jpg' in img_file:
                list1.append(img_file)

    for root, dir, file_dir in os.walk(root2):
        # if file_dir.endswith('.jpg'):
        for img_file in file_dir:
            if 'jpg' in img_file:
                list2.append(img_file)
    
    # print(len(list1), len(list2))
    selected_train_list = random.sample(list1, 200)
    # print(selected_train_list)

    img_list1 = []
    img_list2 = []
    for item in selected_train_list:
        img = cv2.imread(os.path.join(root1, item))
        h,w,c = img.shape
        center_h = h // 2
        center_w = w // 2
        img = img[center_h-44:center_h+44, center_w-44:center_w+44]
        # cv2.imwrite('test8.jpg', cv2.resize(img, (8,8)))
        # print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        img_list1.append(img)
    
    for item in list2:
        img = cv2.imread(os.path.join(root2, item))
        h,w,c = img.shape
        center_h = h // 2
        center_w = w // 2
        img = img[center_h-44:center_h+44, center_w-44:center_w+44]
        # print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        img_list2.append(img)

    # print(len(img_list1))
    # print(len(img_list2))
    similarity = 0
    for img1 in img_list1:
        for img2 in img_list2:
            # similarity += ssim(img1, img2)
            # similarity += mse(img1, img2)
            similarity += classify_gray_hist(img1, img2)
    similarity /= len(img_list1) * len(img_list2)
    return similarity

def main():
    '''
    parser = argparse.ArgumentParser(description='Training efficient networks')
    parser.add_argument('--data', default='./data', help='path to dataset')
    cfg = parser.parse_args()
    for cat in cats:
        train_root = os.path.join(cfg.data, 'TRAIN', cat)
        val_root = os.path.join(cfg.data, 'TEST', cat)
        similarity = calc(train_root, val_root)
        print(cat, similarity)
    '''
    img1 = cv2.imread('avg_train.jpg')
    img2 = cv2.imread('avg_test.jpg')
    img3 = cv2.imread('avg_train_Rotate30.jpg')
    img4 = cv2.imread('avg_Rotate360train.jpg')
    img5 = cv2.imread('avg_train_HF.jpg')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY) 
    img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY) 
    img5 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY) 
    img6 = cv2.imread('avg_train_Rotate10.jpg')
    img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY) 
    sim_R10 = ssim(img6, img2)
    img7 = cv2.imread('avg_105train.jpg')
    img7 = cv2.cvtColor(img7,cv2.COLOR_BGR2GRAY) 
    sim_C = ssim(img7, img2)
    
    sim_base = ssim(img1, img2)
    sim_R30 = ssim(img3, img2)
    sim_R360 = ssim(img4, img2)
    sim_HF = ssim(img5, img2)
    print(sim_base, sim_R10, sim_R30, sim_R360, sim_HF, sim_C)

if __name__ == '__main__':
    main()