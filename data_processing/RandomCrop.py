import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

def Crop(data_folder, save_folder, crop_per_image, out_size):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    trans = transforms.Compose([transforms.CenterCrop(size = out_size+15),
                                transforms.RandomCrop(size = out_size)])
    for folder in os.listdir(data_folder):
        folder_dir = os.path.join(data_folder, folder)
        if os.path.isdir(folder_dir):
            save_dir = os.path.join(save_folder, folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for filename in os.listdir(folder_dir):
                file_dir = os.path.join(folder_dir, filename)
                print('Processing', file_dir)
                for i in range(crop_per_image):
                    file_save_dir = os.path.join(save_dir, filename.split('.')[0]+ '_' + str(i) + '.jpg')
                    print('Saving to', file_save_dir)
                    img = Image.open(file_dir).convert('RGB')
                    img = trans(img)
                    img.save(file_save_dir)
    

def main():
    crop_per_img = 40
    out_size = 88
    data_folder = '/home/SENSETIME/chenzixuan/Desktop/czx/code/data/TRAIN'
    save_folder = '/home/SENSETIME/chenzixuan/Desktop/czx/code/data_crop_105/TRAIN'
    
    Crop(data_folder, save_folder, crop_per_img, out_size)

if __name__ == '__main__':
    main()