##################################################
#
# author: Zixuan Chen
# Date: 2020/01/12
# Description: 
# Training model
#
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import argparse
import os
import time
from data_loader import SARDataset
from Imagenet_dataloader import ImageNetDataset
from lr_scheduler import *
from train_classifier import train_one_epoch
from validate_classifier import validate

# net = models.resnet18(num_classes = 10)
# print(net)

def train(cfg):
    
    data_root = cfg.data
    cls_num = 10
    if data_root == 'imagenet':
        train_dataset = ImageNetDataset()
        val_dataset = ImageNetDataset(False)
        cls_num = 15
    else:
        train_dataset = SARDataset(data_root, inp_size = cfg.img_size)
        val_dataset = SARDataset(data_root, inp_size = cfg.img_size, is_training = False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                                               pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=8)
    
    model_name = cfg.model
    if model_name == 'resnet':
        from modeling.myresnet import ResNet
        net = ResNet(cls_num)
    elif model_name == 'fc':
        from modeling.fc import FCNet
        net = FCNet(cfg.img_size, num_classes=cls_num)
    elif model_name == 'AConv':
        from modeling.AConv import AConv
        net = AConv()
    elif model_name == 'FullyConv_bn':
        from modeling.FullyConv import AConv
        net = AConv(num_classes=cls_num)
    else:
        print(model_name, 'Not Implemeted. Select from resnet18, AConv, FullyConv_bn & fc')
        exit()
    
    best_val_acc = 0
    start_epoch = 0
    if cfg.resume == '':
        pass
    else:
        cpt = torch.load(cfg.resume)
        if 'model_state_dict' in cpt:
            net.load_state_dict(cpt['model_state_dict'])
            # cfg.lr = cpt['lr']
            start_epoch = cpt['epoch'] + 1
            if 'best_acc' in cpt:
                best_val_acc = cpt['best_acc']
        else:
            net.load_state_dict(cpt)
    net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    lr_scheduler = FixedMultiStepLR(cfg.lr, cfg.step)
    
    print(net)
    print('Start Training...')
    print(cfg)
    
    save_dir = cfg.savedir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    log_dir = os.path.join(cfg.savedir, 'log.txt')
    if cfg.save_log:
         f = open(log_dir, 'a')
         print(cfg, file = f)
         f.close()
        
    
    st_time = time.time()
    
        
    for epoch in range(start_epoch, cfg.epochs):
        lr_log = lr_scheduler.step(epoch)
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_log
        
        net.train()
        print("LR for epoch {} = {:.5f}".format(epoch, lr_log))
        train_acc, train_loss, tmp_best_acc = train_one_epoch(
                net, 
                optimizer, 
                train_loader, 
                val_loader,
                epoch, 
                best_val_acc,
                save_dir )
        if tmp_best_acc > best_val_acc:
            best_val_acc = tmp_best_acc
            
        ed_time = time.time()
        
        if (epoch+1)%cfg.ckpt_save_margin == 0:
            model_save_dir = os.path.join(save_dir, 'epoch%d.pth'%(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'lr': lr_log,
                'best_acc': best_val_acc
                }, model_save_dir)
            print('End of epoch %d. Checkpoint has been saved to %s. Epoch Training Time: %f'
                  %(epoch, model_save_dir, ed_time-st_time))
        else:
            print('End of epoch %d. Epoch Training Time: %f'
                  %(epoch, ed_time-st_time))
        
        net.eval()
        val_correct, val_total, val_acc = validate(net, val_loader)
        print('Epoch: %d, Train Accuracy: %f, Val Accuracy: %f'%(epoch, train_acc, val_acc))
        if (val_acc > best_val_acc):
            print('New Best Found!')
            best_val_save_dir = os.path.join(save_dir, 'best_checkpoint')
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_val_save_dir)
            print('Best Model Updated')
        if (cfg.save_log):
            f = open(log_dir, 'a')
            print('Epoch: %d, LR: %f, Train Accuracy: %f, Train Loss: %f, Val Accuracy: %f'%(epoch, lr_log, train_acc, train_loss, val_acc), file = f)
            f.close()
        st_time = time.time()
        

def main():
    parser = argparse.ArgumentParser(description='Training efficient networks')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes to be predicted (default: 10(SAR Dataset))')
    parser.add_argument('--batch-size', default=50, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--data', default='./data', help='path to dataset')
    parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
    # parser.add_argument('--start_epoch', default=0, type=int, help='starting epochs (default: 0)')
    parser.add_argument('--resume', default = '', help='Path to latest checkpoint (default: None)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--model', default='resnet', help='Which model? Support ResNet(resnet), fully Convolution Net(fully_conv) and fully connected(fc).(default: resnet18)')
    # parser.add_argument('--fc_layers', default=3, type=int, help='number of layers for fully_connected net. (default: 3)')
    # parser.add_argument('--fc_channels', default=256, type=int, help='number of channels for each layer of fully_connected net. (default: 256)')
    parser.add_argument('--img_size', default=88, type=int, help='Input size of the image')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=4e-3, type=float, help='weight decay (default: 4e-5)')
    parser.add_argument('--lr-decay', default=0.1, type=float, help='learning rate decay(default: 0.1)')
    parser.add_argument('--step', default=[500, 100], type=int, nargs="+", help='steps at which lr should be decreased.')
    parser.add_argument('--save_log', default=True, help='Set to True to keep record.')
    parser.add_argument('--ckpt_save_margin', default=5, type = int, help='Margin at which ckpts are saved')
    parser.add_argument('--savedir', type=str, default='checkpoint', help='Location to save the results')
    cfg = parser.parse_args()
    
    train(cfg)

if __name__ == '__main__': 
    main()