import numpy as np
import torch
from PIL import Image
import random
import os

def addSaltNoise(pil_img):
    np_img = np.array(pil_img)
    noise_pos = random.randint(100, 1000)
    rows,cols,dims = np_img.shape
    for i in range(noise_pos):
        x=np.random.randint(0,rows)
        y=np.random.randint(0,cols)
        t=random.randint(0,2)
        np_img[x,y,:]=255*t
    new_pil_img = Image.fromarray(np_img.astype('uint8')).convert('RGB')
    return new_pil_img

def addGausianNoise():
    pass

def addRandomUniformNoise():
    pass

def addNoise(data_folder, save_folder, noise_type = 'salt'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for folder in os.listdir(data_folder):
        folder_dir = os.path.join(data_folder, folder)
        if os.path.isdir(folder_dir):
            save_dir = os.path.join(save_folder, folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for filename in os.listdir(folder_dir):
                file_dir = os.path.join(folder_dir, filename)
                print('Processing', file_dir)
                img = Image.open(file_dir).convert('RGB')
                new_save_path = os.path.join(save_dir, filename.split('.')[0]+ '_new.jpg')
                old_save_path = os.path.join(save_dir, filename.split('.')[0]+ '_old.jpg')
                img.save(old_save_path)
                if noise_type == 'salt':
                    new_img = addSaltNoise(img)
                new_img.save(new_save_path)
                
def main():
    data_folder = 'K:\Study\毕设\code\data\TRAIN'
    save_folder = 'K:\Study\毕设\code\salt_noise\TRAIN'
    addNoise(data_folder, save_folder)
    
if __name__ == '__main__':
    main()