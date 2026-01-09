import torch
import torch.nn as nn

from models.module import SeqToANNContainer, pad_and_add, CostCalculator
from models.hooks import count_conv_custom
from neuron import PseudoNeuron


__all__ = ['resnet18', 'resnet19']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, expand=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.expand = expand
        self.conv1 = SeqToANNContainer(
            nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes * expand)
        )
        self.sn1 = PseudoNeuron(**kwargs)
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
        self.sn2 = PseudoNeuron(**kwargs)

    def forward(self, x):
        identity = x
        
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = pad_and_add(out, identity)
        out = self.sn2(out)
        return out


def count_basicblock(m, x, y):
    """Count FLOPs for a BasicBlock."""
    x = x[0][0]
    flops1 = torch.DoubleTensor([0.])
    flops2 = torch.DoubleTensor([0.])
    
    # Count downsample path if exists
    if m.shortcut:
        downsample_flops, _ = count_conv_custom(m.shortcut, x)
        flops1 += downsample_flops

    # Count main path
    flops, x = count_conv_custom(m.conv1, x)  # Remove the () since conv1 is a module, not a function
    flops1 += flops
    flops, _ = count_conv_custom(m.conv2, x)  # Same here
    flops2 += flops

    return [flops1, flops2]


def get_t_basicblock(m, prev_T):
    t1 = m.sn1.get_timesteps()
    t2 = m.sn2.get_timesteps()
        
    return [t1, t2]


custom_hooks = {
    BasicBlock: count_basicblock,
}

custom_timesteps = {
    BasicBlock: get_t_basicblock,
}


class ResNet(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
        )
        self.sn0 = PseudoNeuron(**kwargs)
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2, **kwargs)

        self.avg_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = SeqToANNContainer(nn.Linear(512 * block.expansion, num_classes))

        self.calculator = None

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.conv0(x)
        out = out.unsqueeze(0)
        out = self.sn0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        out = torch.flatten(out, 2)
        out = self.classifier(out).mean(dim=0)
        return out
    
    def forward(self, x):
        if self.training:
            cost = None
            if self.calculator:
                cost = self.calculator.cost()

            return self._forward_impl(x), cost
        else:
            return self._forward_impl(x)
    
    def register_calculator(self, input_size):
        self.calculator = CostCalculator(
            model=self, 
            input_size=input_size,
            custom_hooks=custom_hooks,
            custom_timesteps=custom_timesteps
        )
    
    def verbose(self):
        format_str = "\n" + "="*54 + "\n"
        format_str += "|{:^25}|{:^13}|{:^12}|\n".format("Layer Name", "Scale", "Timesteps")
        format_str += "="*54 + "\n"
        for name, module in self.named_modules():
            if isinstance(module, PseudoNeuron):
                format_str += "|{:^25}|{:^13.3f}|{:^12.1f}|\n".format(
                    name,
                    module.scale.item(),
                    module.get_timesteps().item()
                )
        format_str += "="*54 + "\n"
        return format_str


class ResNet19(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs):
        super(ResNet19, self).__init__()
        self.in_planes = 128
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
        )
        self.sn0 = PseudoNeuron(**kwargs)
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2, **kwargs)
        self.avg_pool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(512 * block.expansion, 256, bias=True)),
            PseudoNeuron(**kwargs),
            SeqToANNContainer(nn.Linear(256, num_classes, bias=True)),
        )

        self.calculator = None

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        if x.ndim == 5: # (T, B, C, H, W)
            out = []
            for t in range(x.shape[0]):
                x_t = self.conv0(x[t])
                x_t = x_t.unsqueeze(0)
                x_t = self.sn0(x_t)
                x_t = self.layer1(x_t)
                x_t = self.layer2(x_t)
                x_t = self.layer3(x_t)
                x_t = self.avg_pool(x_t)
                x_t = torch.flatten(x_t, 2)
                x_t = self.classifier(x_t).mean(dim=0)
                out.append(x_t)
            out = torch.stack(out, dim=0).mean(dim=0)

        else:
            out = self.conv0(x)
            out = out.unsqueeze(0)
            out = self.sn0(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)

            out = torch.flatten(out, 2)
            out = self.classifier(out).mean(dim=0)
        return out

    def forward(self, x):
        if self.training:
            cost = None
            if self.calculator:
                cost = self.calculator.cost()

            return self._forward_impl(x), cost
        else:
            return self._forward_impl(x)
    
    def register_calculator(self, input_size):
        self.calculator = CostCalculator(
            model=self, 
            input_size=input_size,
            custom_hooks=custom_hooks,
            custom_timesteps=custom_timesteps
        )
    
    def verbose(self):
        format_str = "\n" + "="*54 + "\n"
        format_str += "|{:^25}|{:^13}|{:^12}|\n".format("Layer Name", "Scale", "Timesteps")
        format_str += "="*54 + "\n"
        for name, module in self.named_modules():
            if isinstance(module, PseudoNeuron):
                format_str += "|{:^25}|{:^13.3f}|{:^12.1f}|\n".format(
                    name,
                    module.scale.item(),
                    module.get_timesteps().item()
                )
        format_str += "="*54 + "\n"
        return format_str


def resnet18(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channel=in_channel, **kwargs)


def resnet19(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], num_classes, in_channel=in_channel, **kwargs)
