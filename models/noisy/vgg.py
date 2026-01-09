import torch
import torch.nn as nn

from models.module import SeqToANNContainer, CostCalculator
from neuron import PseudoNeuron

__all__ = ['vggsnn_cifar', 'vggsnn_dvs', 'vgg11', 'vgg13']


feature_cfg = {
    'VGG5': [64, 'A', 128, 128, 'A', 'AA'],
    'VGG9': [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512, 'AA'],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512, 'AA'],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFAR': [128, 256, 'A', 512, 'A', 1024, 512],
    'VGGSNN_CIFAR': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'AA3'],
    'VGGSNN_DVS': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'], 
}

clasifier_cfg = {
    'VGG5': [128, 10],
    'VGG11': [512, 10],
    'VGG13': [512, 10],
    'VGG16': [2048, 4096, 4096, 10],
    'VGG19': [2048, 4096, 4096, 10],
    'VGGSNN_CIFAR': [4608, 10],
    'VGGSNN_DVS': [4608, 10]
}


class VGG(nn.Module):
    def __init__(self, architecture='VGG16', kernel_size=3, in_channel=3, use_bias=True,
                 num_class=10, **kwargs):
        super(VGG, self).__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.use_bias = use_bias
        self.num_class = num_class
        clasifier_cfg[architecture][-1] = num_class
        self.feature = self._make_feature(feature_cfg[architecture], **kwargs)
        self.classifier = self._make_classifier(clasifier_cfg[architecture], **kwargs)
        self._initialize_weights()

        self.calculator = None

    def _make_feature(self, config, **kwargs):
        layers = []
        channel = self.in_channel
        for x in config:
            if x == 'A':
                layers.append(SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2)))
            elif x == 'AA':
                layers.append(SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1))))
            elif x == 'AA3':
                layers.append(SeqToANNContainer(nn.AdaptiveAvgPool2d((3, 3))))
            else:
                layers.append(SeqToANNContainer(nn.Conv2d(in_channels=channel, out_channels=x, kernel_size=self.kernel_size,
                                        stride=1, padding=self.kernel_size // 2, bias=self.use_bias)))
                layers.append(SeqToANNContainer(nn.BatchNorm2d(x)))
                layers.append(PseudoNeuron(**kwargs))
                channel = x
        return nn.Sequential(*layers)

    def _make_classifier(self, config, **kwargs):
        layers = []
        for i in range(len(config) - 1):
            layers.append(SeqToANNContainer(nn.Linear(config[i], config[i + 1], bias=self.use_bias)))
            layers.append(PseudoNeuron(**kwargs))
        layers.pop()
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        if x.ndim == 4: # (B, C, H, W)
            x = x.unsqueeze(0)

        x = self.feature(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

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


def vggsnn_cifar(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGGSNN_CIFAR", in_channel=in_channel, num_class=num_classes, **kwargs)

def vggsnn_dvs(num_classes=10, in_channel=2, **kwargs):
    return VGG(architecture="VGGSNN_DVS", in_channel=in_channel, num_class=num_classes, **kwargs)

def vgg11(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG11", in_channel=in_channel, num_class=num_classes, **kwargs)

def vgg13(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG13", in_channel=in_channel, num_class=num_classes, **kwargs)
