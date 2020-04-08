import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(in_channels//4)
        if downsample:
            stride = 2
        else:
            stride = 1
        self.conv2 = nn.Conv2d(in_channels//4, in_channels//4, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.conv3 = nn.Conv2d(in_channels//4, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.down_sample = None
        if downsample:
            self.down_sample = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
            self.bn_downsample = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        branch2 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.down_sample != None:
            branch2 = self.down_sample(branch2)
            branch2 = self.bn_downsample(branch2)
        
        out = x + branch2
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn = nn.BatchNorm2d(32)
        
        self.stage1_1 = Bottleneck(32, 32)
        self.stage1_2 = Bottleneck(32, 64, True)
        
        self.stage2_1 = Bottleneck(64, 64)
        # self.stage2_2 = Bottleneck(64, 64)
        self.stage2_2 = Bottleneck(64, 128, True)
        
        self.stage3_1 = Bottleneck(128, 128)
        # self.stage3_2 = Bottleneck(128, 128)
        self.stage3_2 = Bottleneck(128, 256, True)
        
        self.stage4_1 = Bottleneck(256,256)
        self.stage4_2 = Bottleneck(256, 512, True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 256)
        self.relu = nn.ReLU(inplace = True)
        self.classification = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        # x = self.stage2_3(x)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        # x = self.stage3_3(x)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.classification(out)
        return out