import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

def Rotate(data_folder, save_folder, out_size):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # trans_crop = transforms.CenterCrop(size = out_size)
    # trans_hori_flip = transforms.RandomHorizontalFlip(p=1)
    # trans_verti_flip = transforms.RandomVerticalFlip(p=1)
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
                for i in range(0, 360, 45):
                    file_save_dir = os.path.join(save_dir, filename.split('.')[0]+ '_' + str(i) + '.jpg')
                    new_img = img.rotate(i)
                    new_img.save(file_save_dir)
                    '''
                    if i != 0:
                        file_save_dir = os.path.join(save_dir, filename.split('.')[0]+ '_' + str(360-i) + '.jpg')
                        new_img = img.rotate(360-i)
                        new_img.save(file_save_dir)
                    '''
                    
                
def main():
    # crop_per_img = 10
    out_size = 88
    data_folder = '/home/SENSETIME/chenzixuan/Desktop/czx/code/data_small_set/TRAIN'
    save_folder = '/home/SENSETIME/chenzixuan/Desktop/czx/code/data_small_Rotate360_45degree/TRAIN'
    
    Rotate(data_folder, save_folder, out_size)

if __name__ == '__main__':
    main()
