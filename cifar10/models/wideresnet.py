# https://github.com/uoguelph-mlrg/Cutout

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut: x   = self.relu1(self.bn1(x))
        else:                   out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x): return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        layers = [nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)]
        layers.append( NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate))
        layers.append( NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate))
        layers.append( NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate))
        layers += [nn.BatchNorm2d(nChannels[3]), nn.ReLU(inplace=True), AdaptiveConcatPool2d(1), Flatten(),
                   nn.Linear(2*nChannels[3], num_classes)]
        self.features = nn.Sequential(*layers)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): m.bias.data.zero_()

    def forward(self, x): return self.features(x)

def wrn_16_1(): return WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.)
def wrn_22(): return WideResNet(depth=22, num_classes=10, widen_factor=6, dropRate=0.)
def wrn_22_k8(): return WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.)
def wrn_22_k10(): return WideResNet(depth=22, num_classes=10, widen_factor=10, dropRate=0.)
def wrn_22_k8_p2(): return WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.2)
def wrn_28(): return WideResNet(depth=28, num_classes=10, widen_factor=6, dropRate=0.)
def wrn_28_k8(): return WideResNet(depth=28, num_classes=10, widen_factor=8, dropRate=0.)
def wrn_28_k8_p2(): return WideResNet(depth=28, num_classes=10, widen_factor=8, dropRate=0.2)
def wrn_28_p2(): return WideResNet(depth=28, num_classes=10, widen_factor=6, dropRate=0.2)


