import torch
import torch.nn as nn
from torchvision import models

from torchprofiler import Profiler

model = models.resnet50()

profiler = Profiler(model, caffe_style = True)
profiler.run((1,3,224,224))
profiler.print_summary()
