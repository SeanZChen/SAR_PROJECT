import torch
import torch.nn as nn
import torch.nn.functional as F

class AConv(nn.Module):
    def __init__(self):
        super(AConv, self).__init__()
        # self.convnet = nn.ModuleList()
        self.conv1 = nn.Conv2d(3, 16, 5, 1, 0)
        self.pooler1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.pooler2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 6, 1, 0)
        self.pooler3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 0)
        self.dropout = nn.Dropout(p=0.2)
        self.conv_out = nn.Conv2d(128, 10, 3, 1, 0)
        '''
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_out]:
            nn.init.normal_(layer.weight, mean=0, std = 0.01) #print(layer)
            nn.init.constant_(layer.bias, 0.1)
        '''
    
    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.size())
        x = self.pooler1(x)
        # print(x.size())
        x = self.conv2(x)
        x = F.relu(x)
        
        # print(x.size())
        x = self.pooler2(x)
        # print(x.size())
        x = self.conv3(x)
        x = F.relu(x)
        # print(x.size())
        x = self.pooler3(x)
        # print(x.size())
        x = self.conv4(x)
        x = F.relu(x)
        # print(x.size())
        x = self.dropout(x)
        x = self.conv_out(x)
        # logits = F.adaptive_avg_pool2d(x, (1, 1))
        # print(logits.size())
        logits = x.view(x.size(0), -1)
        # print(logits.size())
        # print(logits)
        return logits
        
        
            