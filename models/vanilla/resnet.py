import torch
import torch.nn as nn
from functools import partial

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.layer import SeqToANNContainer
from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode


__all__ = ['resnet18', 'resnet19']


Neuron = partial(MultiStepLIFNode, tau=2.0, v_reset=None, v_threshold=1.0, decay_input=False, detach_reset=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, expand=1):
        super(BasicBlock, self).__init__()
        self.expand = expand
        self.conv1 = SeqToANNContainer(
            nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes * expand)
        )
        self.sn1 = MultiStepLIFNode()
        self.conv2 = SeqToANNContainer(
            nn.Conv2d(planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * expand)
        )
        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = SeqToANNContainer(
                nn.Conv2d(in_planes, planes * self.expansion * expand, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes * expand)
            )
        self.sn2 = MultiStepLIFNode()

    def forward(self, x):
        identity = x
        
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = out + identity
        out = self.sn2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, T=4):
        super(ResNet, self).__init__()
        self.T = T
        self.in_planes = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
        )
        self.sn0 = Neuron()
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2)

        self.avg_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = SeqToANNContainer(nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):        
        out = self.conv0(x)
        out = out.unsqueeze(0)
        out = out.repeat(self.T, 1, 1, 1, 1)
        
        out = self.sn0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        out = torch.flatten(out, 2)
        out = self.classifier(out).mean(dim=0)
        return out
    

class ResNet19(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, T=4):
        super(ResNet19, self).__init__()
        self.T = T
        self.in_planes = 128
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
        )
        self.sn0 = Neuron()
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2)
        self.avg_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(512 * block.expansion, 256, bias=True)),
            Neuron(),
            SeqToANNContainer(nn.Linear(256, num_classes, bias=True)),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        functional.reset_net(self)
        
        out = self.conv0(x)
        out = out.unsqueeze(0)
        out = out.repeat(self.T, 1, 1, 1, 1)
        
        out = self.sn0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)

        out = torch.flatten(out, 2)
        out = self.classifier(out).mean(dim=0)
        return out


def resnet18(num_classes=10, in_channel=3, T=4, neuron_dropout=0.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channel=in_channel, T=T)


def resnet19(num_classes=10, in_channel=3, T=4, neuron_dropout=0.0):
    return ResNet19(BasicBlock, [3, 3, 2], num_classes, in_channel=in_channel, T=T)
