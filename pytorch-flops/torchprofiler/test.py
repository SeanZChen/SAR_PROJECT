from functools import partial
import torch

CustomConv2d = partial(torch.nn.Conv2d, bias=True)

conv1=CustomConv2d(3,8,3)
from torchprofiler import Profiler

profiler = Profiler(conv1)
profiler.run((1,3,224,224))
profiler.print_summary()
