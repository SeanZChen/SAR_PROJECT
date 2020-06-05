# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:21:29 2019

@author: czx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AConv(nn.Module):
    def __init__(self, num_classes=10):
        super(AConv, self).__init__()
        # self.convnet = nn.ModuleList()
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pooler1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pooler2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pooler3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pooler4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 5, 1, 2)
        self.bn5 = nn.BatchNorm2d(256)
        # self.dropout = nn.Dropout(p=0.2)
        self.conv_out = nn.Conv2d(256, num_classes, 5, 1, 2)
        self.pooler5 = nn.AdaptiveMaxPool2d((1,1))
        '''
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_out]:
            nn.init.normal_(layer.weight, mean=0, std = 0.01) #print(layer)
            nn.init.constant_(layer.bias, 0.1)
        '''
    
    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # print(x.size())
        x = self.pooler1(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # print(x.size())
        x = self.pooler2(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # print(x.size())
        x = self.pooler3(x)
        # print(x.size())
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pooler4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        # print(x.size())
        # x = self.dropout(x)
        # x = self.conv_out(x)
        x = self.pooler5(x)
        # print(x.size())
        logits = x.view(x.size(0), -1)
        # print(logits.size())
        # print(logits)
        return logits
        
        
            