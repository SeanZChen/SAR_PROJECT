import os
import cv2
import random


def get_all_files(root):

    filename_list = []
    for root, dirs, files in os.walk(root):
        for filename in files:
            if filename.endswith('.jpg'):
                filename_list.append(os.path.join(root,filename))
    return filename_list
    
clses = ['aircraft', 'birds', 'cars', 'dogs', 'flowers']
save_root = './selected'

for cls in clses:
    imgs = get_all_files(cls)
    selected_imgs = random.sample(imgs, 5)
    
    for img in selected_imgs:
        # print(os.path.join(cls, img))
        image = cv2.imread(img)
        print(image.shape)
        image = cv2.resize(image, (128,128))
        save_dir = os.path.join(save_root, img.split('/')[-1])
        cv2.imwrite(save_dir, image)
