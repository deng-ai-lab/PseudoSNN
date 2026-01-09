import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


def l_prod(in_list):
    """Compute the product of all elements in the input list."""
    res = 1
    for _ in in_list:
        res *= _
    return res


def l_sum(in_list):
    """Calculate the sum of all numerical elements in a list."""
    return sum(in_list)


def calculate_zero_ops():
    """Initializes and returns a tensor with all elements set to zero."""
    return torch.DoubleTensor([0])


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    """Calculate FLOPs for a Conv2D layer using input/output sizes, kernel size, groups, and the bias flag."""
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def calculate_norm(input_size):
    """Compute the L2 norm of a tensor or array based on its input size."""
    return torch.DoubleTensor([2 * input_size])


def calculate_avgpool(input_size):
    """Calculate the average pooling size for a given input tensor."""
    return torch.DoubleTensor([int(input_size)])


def calculate_adaptive_avg(kernel_size, output_size):
    """Calculate FLOPs for adaptive average pooling given kernel size and output size."""
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for given input feature and number of elements."""
    return torch.DoubleTensor([int(in_feature * num_elements)])


def zero_ops(m, x, y):
    """Incrementally add zero operations to the model's total operations count."""
    return calculate_zero_ops()


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    """Calculate and add the number of convolutional operations (FLOPs) for a ConvNd layer to the model's total ops."""
    x = x[0]

    return calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
    )


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    """Calculate and add the FLOPs for a batch normalization layer, including elementwise and affine operations."""
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if getattr(m, "affine", False) or getattr(m, "elementwise_affine", False):
        flops *= 2
    return flops


def count_avgpool(m, x, y):
    """Calculate and update the total number of operations (FLOPs) for an AvgPool layer based on the output elements."""
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    return calculate_avgpool(num_elements)


def count_adap_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AdaptiveAvgPool layer using kernel and element counts."""
    kernel = torch.div(torch.DoubleTensor([*(x[0].shape[2:])]), torch.DoubleTensor([*(y.shape[2:])]))
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    return calculate_adaptive_avg(total_add, num_elements)


def count_linear(m, x, y):
    """Counts total operations for nn.Linear layers using input and output element dimensions."""
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    return calculate_linear(total_mul, num_elements)



def count_conv_custom(modules, x):
    """Helper function to count FLOPs for conv+bn sequence"""
    flops = torch.DoubleTensor([0.])
    for module in modules:
        if isinstance(module, nn.Conv2d):
            y = F.conv2d(x, module.weight, module.bias, 
                        stride=module.stride, padding=module.padding,
                        dilation=module.dilation, groups=module.groups)
            flops += count_convNd(module, x, y)
            x = y
        # elif isinstance(module, nn.BatchNorm2d):
        #     flops += count_normalization(module, x, y)
        #     x = y
    return flops, x


register_hooks = {
    nn.Conv2d: count_convNd,
    # The homogeneity of convolution allows the following BN and linear scaling transformation
    # to be equivalently fused into the convolutional layer with an added bias when deployment. 
    # Therefore, when calculating the energy consumption,the consumption of BN layers could be ignored.
    # nn.BatchNorm2d: count_normalization,
    nn.Linear: count_linear,
    nn.MaxPool2d: zero_ops,
    nn.AvgPool2d: count_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
}