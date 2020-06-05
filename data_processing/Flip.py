import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

def Flip(data_folder, save_folder, out_size):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    trans_crop = transforms.CenterCrop(size = out_size)
    trans_hori_flip = transforms.RandomHorizontalFlip(p=1)
    trans_verti_flip = transforms.RandomVerticalFlip(p=1)
    for folder in os.listdir(data_folder):
        folder_dir = os.path.join(data_folder, folder)
        if os.path.isdir(folder_dir):
            save_dir = os.path.join(save_folder, folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for filename in os.listdir(folder_dir):
                file_dir = os.path.join(folder_dir, filename)
                print('Processing', file_dir)
                file_save_dir1 = os.path.join(save_dir, filename.split('.')[0]+ '_1.jpg')
                file_save_dir2 = os.path.join(save_dir, filename.split('.')[0]+ '_2.jpg')
                file_save_dir3 = os.path.join(save_dir, filename.split('.')[0]+ '_3.jpg')
                file_save_dir4 = os.path.join(save_dir, filename.split('.')[0]+ '_4.jpg')
                img = Image.open(file_dir).convert('RGB')
                img1 = img
                img2 = trans_hori_flip(img1)
                img3 = trans_verti_flip(img1)
                img4 = trans_verti_flip(img2)
                img1.save(file_save_dir1)
                img2.save(file_save_dir2)
                img3.save(file_save_dir3)
                img4.save(file_save_dir4)
                
    

def main():
    # crop_per_img = 10
    out_size = 88
    data_folder = 'K:\Study\毕设\code\data\TRAIN'
    save_folder = 'K:\Study\毕设\code\data_flip\TRAIN'
    
    Flip(data_folder, save_folder, out_size)

if __name__ == '__main__':
    main()