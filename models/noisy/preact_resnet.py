import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List, Optional, Type, Union
from torch.hub import load_state_dict_from_url

from models.module import SeqToANNContainer, pad_and_add, CostCalculator
from models.hooks import count_conv_custom
from neuron import PseudoNeuron

__all__ = ['preact_resnet34', ]

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
}

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            neuron: callable = None,
            **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn1 = neuron(**kwargs)
        self.conv1 = SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn2 = neuron(**kwargs)
        self.conv2 = SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.sn1(x)
        out = self.conv1(out)

        out = self.sn2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = pad_and_add(out, identity)
        return out


def count_basicblock(m, x, y):
    """Count FLOPs for a BasicBlock."""
    x = x[0][0]
    flops0 = torch.DoubleTensor([0.])
    flops1 = torch.DoubleTensor([0.])
    flops2 = torch.DoubleTensor([0.])
    
    # Count downsample path if exists
    if m.downsample:
        downsample_flops, _ = count_conv_custom(m.downsample, x)
        flops0 += downsample_flops

    # Count main path
    flops, x = count_conv_custom(m.conv1, x)  # Remove the () since conv1 is a module, not a function
    flops1 += flops
    flops, _ = count_conv_custom(m.conv2, x)  # Same here
    flops2 += flops

    return [flops0, flops1, flops2]


def get_t_basicblock(m, prev_T):
    t1 = m.sn1.get_timesteps()
    t2 = m.sn2.get_timesteps()
    
    t_final = torch.max(t2, prev_T)
        
    return [t1, t2, t_final]


custom_hooks = {
    BasicBlock: count_basicblock,
}

custom_timesteps = {
    BasicBlock: get_t_basicblock,
}


class PreactSpikingResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, ]],
            layers: List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            neuron: callable = None,
            **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       neuron=neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       neuron=neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       neuron=neuron, **kwargs)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.calculator = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, )):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
            self,
            block: Type[Union[BasicBlock, ]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            neuron: callable = None,
            **kwargs
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SeqToANNContainer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                neuron, **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    neuron=neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        out.unsqueeze_(0)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 2)
        return self.fc(out.mean(dim=0))
    

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


def spiking_resnet(arch, block, layers, pretrained, progress, neuron, **kwargs):
    model = PreactSpikingResNet(block, layers, neuron=neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def preact_resnet34(pretrained=False, progress=True, neuron: callable = PseudoNeuron, **kwargs):
    return spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                                  neuron=PseudoNeuron, **kwargs)

