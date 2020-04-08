##################################################
#
# author: Zixuan Chen
# Date: 2020/04/07
# Description: 
# Sample small training set
#
##################################################

import os
import random
import cv2
import argparse
cats = ['2S1', 'BRDM_2', 'BTR_60', 'D7', 'BMP2', 'BTR70', 'T72', 'T62', 'ZIL131', 'ZSU_23_4']
def sample(cfg):
    if not os.path.exists(cfg.save_root):
        os.makedirs(cfg.save_root)
    else:
        os.system("rm -rf %s"%(cfg.save_root))
    for cat in cats:
        original_dir = os.path.join(cfg.original_root, 'TRAIN', cat)
        save_dir = os.path.join(cfg.save_root, 'TRAIN', cat)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imgs = os.listdir(original_dir)
        sample_size = min(len(imgs), cfg.sample_size)
        print('Sampleing %s, sampled: %d'%(cat, sample_size))
        selected = random.sample(imgs, sample_size)
        for item in selected:
            selected_dir = os.path.join(original_dir, item)
            # new_dir = os.path.join(save_dir, item)
            os.system("cp %s %s"%(selected_dir, save_dir))
    os.system("cp -r %s %s"%(os.path.join(cfg.original_root, 'TEST'), cfg.save_root))


def main():
    parser = argparse.ArgumentParser(description='Training efficient networks')
    parser.add_argument('--sample-size', default=50, type=int)
    parser.add_argument('--original_root', default='./data', type=str)
    parser.add_argument('--save_root', default='./data_small_set', type=str)
    cfg = parser.parse_args()
    sample(cfg)

if __name__ == '__main__':
    main()
