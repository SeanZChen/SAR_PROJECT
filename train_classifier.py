# ============================================
__author__ = "CHEN Zixuan"
__maintainer__ = "CHEN Zixuan"
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from validate_classifier import validate
import os

def calc_acc(out, target):
    ans = torch.argmax(out, 1)
    correct = (ans == target).float().sum()
    acc = correct/(target.size(0))
    return acc

def calc_mean(l):
    sum = 0
    for item in l:     
        sum += item  
    return sum/len(l)

def train_one_epoch(model, optimizer, train_dataloader, val_dataloader, epoch, best_acc = 0, save_dir = ''):
    model.train()
    acc = []
    loss = []
    log_dir = os.path.join(save_dir, 'log.txt')
    log = open(log_dir,  'a')
    for i, (input, target) in enumerate(train_dataloader):
        input = input.cuda()
        target = target.cuda()
        out = model(input)
        new_loss = F.cross_entropy(out, target)
        new_acc = calc_acc(out, target)
        acc.append(new_acc)
        loss.append(new_loss.item())
        
        if (i+1)%10 == 0:
            print("epoch: %d, iter: %d, loss: %f(avg: %f), acc: %f(avg: %f)"%
                  (epoch, i+1, new_loss.item(), calc_mean(loss), new_acc.item(), calc_mean(acc)))
        if i%500 == 0 and i!= 0:
            print('Validating...')
            val_correct, val_total, val_acc = validate(model, val_dataloader)
            print('Val Acc: %f'%(val_acc))
            if val_acc > best_acc:
                print('New Best Found')
                best_val_save_dir = os.path.join(save_dir, 'best_checkpoint.pth')
                torch.save(model.state_dict(), best_val_save_dir)
                best_acc = val_acc
                print('Saving New Best Result to', best_val_save_dir)
            print('epoch: %d, iter: %d, val acc: %f'%(epoch, i, val_acc), file = log)
            tmp_save_dir = os.path.join(save_dir, 'epoch%d_iter%d.pth'%(epoch, i))
            print('Saving Checkpoint to', tmp_save_dir)
            torch.save(model.state_dict(), tmp_save_dir)
        
        optimizer.zero_grad()
        new_loss.backward()
        optimizer.step()
    log.close()
                            
    return calc_mean(acc), calc_mean(loss), best_acc
        