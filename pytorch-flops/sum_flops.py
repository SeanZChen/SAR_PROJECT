import torch
import torch.nn as nn

from torchprofiler import Profiler
from modeling.FullyConv import AConv
from modeling.fc import FCNet
from modeling.myresnet import ResNet

import torchvision.models as models
import torchvision.models.resnet as resnet

data = (1,3,88,88)
## resnet50_classifier
#model = models.resnet50()
#import enet_official_dw4 as model
#model = model.ENet(19,False,False)
# print(dir(resnet))
# net = FCNet(in_resolution = 88)
net = AConv()
# net = ResNet(10)
# net = resnet.ResNet(resnet.Bottleneck, [2,2,1,1], num_classes=10)
profiler = Profiler(net, caffe_style = True)
profiler.run(data)
profiler.print_summary()
