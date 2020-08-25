"""SE-ResNet in PyTorch
Based on preact_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ['SPAResNet18', 'SPAResNet34', 'SPAResNet50', 'SPAResNet101', 'SPAResNet152']

class SPALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = self.scale = Parameter(torch.ones(1,3,1,1,1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel//reduction,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2,scale_factor=2).unsqueeze(dim=1),
             F.interpolate(y1,scale_factor=4).unsqueeze(dim=1)],
            dim=1
        )
        y = (y*self.weight).sum(dim=1,keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y,size = x.size()[2:])
        y = y.expand_as(x)
        return x*y



class PreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.spa = SPALayer(planes,reduction)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.spa(out)
        out += shortcut
        return out


class PreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(PreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.spa = SPALayer(self.expansion*planes, reduction)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.spa(out)
        out +=shortcut
        return out


class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10,reduction=16):
        super(ResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction)
        self.linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self,block, planes, num_blocks,stride,reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SPAResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2],num_classes)


def SPAResNet34(num_classes=10):
    return ResNet(PreActBlock, [3,4,6,3],num_classes)


def SPAResNet50(num_classes=10):
    return ResNet(PreActBootleneck, [3,4,6,3],num_classes)


def SPAResNet101(num_classes=10):
    return ResNet(PreActBootleneck, [3,4,23,3],num_classes)


def SPAResNet152(num_classes=10):
    return ResNet(PreActBootleneck, [3,8,36,3],num_classes)



def demo():
    net = SPAResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())


demo()
