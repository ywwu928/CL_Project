import torch.nn as nn 
from low_precision_utils import * 

class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.conv1 = SConv2d(1, 16, kernel_size=3, stride=1, 
                               padding=1)
        self.bn1   = SBatchNorm(16)
        self.act1  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = SConv2d(16, 32, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2   = SBatchNorm(32)
        self.act2  = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2  = SLinear(7*7*32, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        # print(c1.shape) 
        b1  = self.bn1(c1)
        # print(b1.shape) 
        a1  = self.act1(b1)
        # print(a1.shape) 
        p1  = self.pool1(a1)
        # print(p1.shape) 
        c2  = self.conv2(p1)
        # print(c2.shape) 
        b2  = self.bn2(c2)
        # print(b2.shape) 
        a2  = self.act2(b2)
        # print(a2.shape) 
        p2  = self.pool2(a2)
        # print(p2.shape) 
        flt = p2.view(p2.size(0), -1)
        # print(flt.shape) 
        out = self.lin2(flt)
        return out