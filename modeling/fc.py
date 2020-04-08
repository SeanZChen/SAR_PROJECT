import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, in_resolution = 100, in_channels = 3, num_classes = 10):
        super(FCNet, self).__init__()
        net_channels = [64, 128, 128, 256, 256, 256]
        layers = 6
        channels = in_channels * (in_resolution ** 2)
        fcnet = nn.ModuleList()
        for i in range(layers):
            fcnet.append(nn.Linear(channels, net_channels[i]))
            # fcnet.add_module('relu%d'%i, nn.ReLU())
            channels = net_channels[i]
        
        self.feature_extract = fcnet
        self.cls_score = nn.Linear(channels, num_classes)
        '''
        for layer in self.feature_extract:
            nn.init.normal_(layer.weight, mean=0, std = 0.01) #print(layer)
            nn.init.constant_(layer.bias, 0.1)
        '''
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.feature_extract:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2)
        # feature = self.feature_extract(x)
        cls_logits = self.cls_score(x)
        return cls_logits
        
            