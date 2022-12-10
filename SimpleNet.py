import torch.nn as nn 
from low_precision_utils import * 

class MyConvNet(nn.Module):
    def __init__(self, args):
        super(MyConvNet, self).__init__()
        self.conv1 = SConv2d(1, 16, kernel_size=3, stride=1, 
                               padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = SConv2d(16, 32, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.act2  = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2  = SLinear(mf, 7*7*32, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        b1  = self.bn1(c1)
        a1  = self.act1(b1)
        p1  = self.pool1(a1)
        c2  = self.conv2(p1)
        b2  = self.bn2(c2)
        a2  = self.act2(b2)
        p2  = self.pool2(a2)
        flt = p2.view(p2.size(0), -1)
        out = self.lin2(flt)
        return out