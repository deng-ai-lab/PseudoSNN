import torch
import torch.nn as nn

from models.module import SeqToANNContainer, pad_and_add, CostCalculator
from models.hooks import count_conv_custom
from neuron import PseudoNeuron


__all__ = [
    'sew_resnet19', 'sew_resnet20', # CIFAR
    'sew_resnet18', 'sew_resnet34', 'sew_resnet50' # ImageNet
]


NEURON = PseudoNeuron


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = NEURON(**kwargs)

        self.conv2 = SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = NEURON(**kwargs)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))

        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            # out = out + identity
            out = pad_and_add(out, identity)
        elif self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = NEURON(**kwargs)

        self.conv2 = SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = NEURON(**kwargs)

        self.conv3 = SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = NEURON(**kwargs)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))

        out = self.sn2(self.conv2(out))

        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.connect_f == 'ADD':
            # out = out + identity
            out = pad_and_add(out, identity)
        elif self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv3[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2[1].bias, 1)


def count_sew_basicblock(m, x, y):
    """Count FLOPs for a BasicBlock."""
    x = x[0][0]
    flops1 = torch.DoubleTensor([0.])
    flops2 = torch.DoubleTensor([0.])
    
    # Count downsample path if exists
    if m.downsample:
        downsample_flops, _ = count_conv_custom(m.downsample, x)
        flops1 += downsample_flops

    # Count main path
    flops, x = count_conv_custom(m.conv1, x)  # Remove the () since conv1 is a module, not a function
    flops1 += flops
    flops, _ = count_conv_custom(m.conv2, x)  # Same here
    flops2 += flops
    return [flops1, flops2]


def count_sew_bottleneck(m, x, y):
    """Count FLOPs for a Bottleneck."""
    x = x[0][0]
    flops1 = torch.DoubleTensor([0.])
    flops2 = torch.DoubleTensor([0.])
    flops3 = torch.DoubleTensor([0.])
    
    # Count downsample path if exists
    if m.downsample:
        downsample_flops, _ = count_conv_custom(m.downsample, x)
        flops1 += downsample_flops

    # Count main path
    flops, x = count_conv_custom(m.conv1, x)  # Remove the () here too
    flops1 += flops
    flops, _ = count_conv_custom(m.conv2, x)
    flops2 += flops
    flops, _ = count_conv_custom(m.conv3, x)
    flops3 += flops
    
    return [flops1, flops2, flops3]


def get_t_sew_basicblock(m, prev_T):
    t1 = m.sn1.get_timesteps()
    t2 = m.sn2.get_timesteps()
    
    # Get timesteps considering downsample path
    if m.downsample:
        t_downsample = m.downsample[1].get_timesteps()
        t_final = torch.max(t2, t_downsample)
    else:
        t_final = torch.max(t2, prev_T)
        
    return [t1, t_final]


def get_t_sew_bottleneck(m, prev_T):
    t1 = m.sn1.get_timesteps()
    t2 = m.sn2.get_timesteps()
    t3 = m.sn3.get_timesteps()
    
    # Get timesteps considering downsample path
    if m.downsample:
        t_downsample = m.downsample[1].get_timesteps()
        t_final = torch.max(t3, t_downsample)
    else:
        t_final = torch.max(t3, prev_T)
        
    return [t1, t2, t_final]


custom_hooks = {
    BasicBlock: count_sew_basicblock,
    Bottleneck: count_sew_bottleneck,
}

custom_timesteps = {
    BasicBlock: get_t_sew_basicblock,
    Bottleneck: get_t_sew_bottleneck,
}


class SEWResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, connect_f='ADD', **kwargs):
        super(SEWResNet, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)


        self.sn1 = NEURON(**kwargs)
        self.maxpool = SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f, **kwargs)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.calculator = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                NEURON(**kwargs)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return self.fc(x.mean(dim=0))

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


def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


##################################################################################
################################ ResNet for CIFAR ################################
##################################################################################

class ResNet19(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, connect_f='ADD', **kwargs):
        super(ResNet19, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False) #we use a different inputsize than the original paper
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = NEURON(**kwargs)

        self.layer1 = self._make_layer(block, 128, layers[0], connect_f=connect_f, **kwargs)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f, **kwargs)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f, **kwargs)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(
            SeqToANNContainer(nn.Linear(512 * block.expansion, 256)),
            NEURON(**kwargs),
            SeqToANNContainer(nn.Linear(256, num_classes)),
        )

        self.calculator = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                NEURON(**kwargs)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0)
        
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        output = self.fc(x).mean(dim=0)
        return output

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


class ResNet20(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, connect_f='ADD', **kwargs):
        super(ResNet20, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.skip = ['conv1', 'fc']

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False) #we use a different inputsize than the original paper
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = NEURON(**kwargs)

        self.layer1 = self._make_layer(block, 16, layers[0], connect_f=connect_f, **kwargs)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f, **kwargs)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f, **kwargs)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(
            SeqToANNContainer(nn.Linear(64 * block.expansion, num_classes)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)
        
        self.calculator = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                NEURON(**kwargs)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(0)
        
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        output = self.fc(x).mean(dim=0)
        return output

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


def sew_resnet19(**kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], **kwargs)


def sew_resnet20(**kwargs):
    return ResNet20(BasicBlock, [3, 3, 3], **kwargs)
