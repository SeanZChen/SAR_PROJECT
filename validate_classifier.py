# ============================================
__author__ = "CHEN Zixuan"
__maintainer__ = "CHEN Zixuan"
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_acc(out, target):
    ans = torch.argmax(out, 1)
    correct = (ans == target).float().sum()
    acc = correct/(target.size(0))
    return acc, correct

def calc_mean(l):
    sum = 0
    for item in l:     
        sum += item  
    return sum/len(l)

def validate(model, dataloader):
    model.eval()
    total_correct = 0
    total = 0
    for i, (input, target) in enumerate(dataloader):
        input = input.cuda()
        target = target.cuda()
        out = model(input)
        # new_loss = F.cross_entropy(out, target)
        new_acc, correct = calc_acc(out, target)
        total_correct += correct
        total += target.size(0)
        # loss.append(new_loss.item())
    model.train()
    
    return total_correct, total, total_correct/total
        
        