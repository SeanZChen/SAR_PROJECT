import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchprofiler import Profiler
from functools import partial

def run_profiler(input_shape, model):
    profiler = Profiler(model)
    profiler.run(input_shape)
    return profiler
    
class Conv2dCustomBias(nn.Module):
    def __init__(self):
        super(Conv2dCustomBias, self).__init__()
        self.conv = nn.Conv2d(3, 8, (3, 3),1, 1, bias = False)
    
    def forward(self,x, y, k = 1):
        x = self.conv(x)
        return k * x + y

class Conv2dWithOtherParent(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dWithOtherParent, self).__init__(*args, **kwargs)

class TestProfiler(unittest.TestCase):
    
    def test_ops_conv2d(self):
        input_shape = (1, 3, 224, 224)
        model = nn.Conv2d(3, 8, (3, 3), 1, 1, bias = False)

        # 224 * 224 * 3 * 8 * 3 * 3 * 2
        self.assertEqual(run_profiler(input_shape, model).total_flops, 21676032)

    def test_ops_convtransposed2d(self):
        input_shape = (1, 3, 224, 224)
        model = nn.ConvTranspose2d(3,19,kernel_size = 2,stride = 2, groups = 1, bias=False)
        # 224 * 224 * 3 * 19 * 2 * 2 * 2
        self.assertEqual(run_profiler(input_shape, model).total_flops, 22880256)

    def test_ops_conv3d(self):
        input_shape = (1, 3, 112, 112, 112)
        model = nn.Conv3d(3, 8, (3, 3, 3), 1, 1, bias = False)

        # 224**3 * 3 * 8 * 3**3 * 2
        self.assertEqual(run_profiler(input_shape, model).total_flops, 1820786688)

    def test_ops_linear(self):
        input_shape = (1, 256)
        model = nn.Linear(256, 1000)
        # 256 * 1000 * 2 + 1000
        self.assertEqual(run_profiler(input_shape, model).total_flops, 513000)

    def test_multi_input(self):
        input_shape = [(1, 3, 224, 224), (1, 8, 224, 224)]
        model = Conv2dCustomBias()
        profiler = Profiler(model)
        profiler.run(*input_shape)
        self.assertEqual(profiler.total_flops, 21676032)

    def test_run_with_input(self):
        x = torch.rand(1, 3, 224, 224)
        y = torch.rand(1, 8, 224, 224)
        k = 2.0
        model = Conv2dCustomBias()
        profiler = Profiler(model)
        profiler.run_with_input(x, y, k)
        self.assertEqual(profiler.total_flops, 21676032)

    def test_ops_conv2d_with_other_parent(self):
        input_shape = (1, 3, 224, 224)
        model = Conv2dWithOtherParent(3, 8, (3, 3), 1, 1, bias = False)

        # 224 * 224 * 3 * 8 * 3 * 3 * 2
        self.assertEqual(run_profiler(input_shape, model).total_flops, 21676032)

    def test_ops_conv2d_partial(self):
        input_shape = (1, 3, 224, 224)
        CustomConv2d = partial(nn.Conv2d, bias = False)
        model = CustomConv2d(3, 8, (3, 3), 1, 1)

        # 224 * 224 * 3 * 8 * 3 * 3 * 2
        self.assertEqual(run_profiler(input_shape, model).total_flops, 21676032)

        

if __name__ == '__main__':
    unittest.main()
