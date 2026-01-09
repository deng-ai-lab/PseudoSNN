import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict, Callable, Set

from models.hooks import register_hooks
from neuron import PseudoNeuron

import torch
import torch.nn.functional as F

def pad_and_add(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Adds two tensors with shapes [T1, B, ...] and [T2, B, ...] after padding
    the shorter tensor along the first dimension (dim=0) using mirror reflection padding.

    Args:
        tensor1: The first input tensor [T1, B, ...].
        tensor2: The second input tensor [T2, B, ...].

    Returns:
        A new tensor resulting from the element-wise addition of the padded
        tensors. Shape: [max(T1, T2), B, ...].
    """
    # If lengths are equal, no padding is needed.
    if tensor1.shape[0] == tensor2.shape[0]:
        return tensor1 + tensor2

    # Sort the tensors by length to identify the shorter and longer one.
    t_short, t_long = sorted([tensor1, tensor2], key=lambda t: t.shape[0])

    # Calculate the padding length needed.
    diff = t_long.shape[0] - t_short.shape[0]

    # Manual implementation of 1D reflection padding
    if diff > 0:
        # Create reflection indices
        indices = torch.arange(t_short.shape[0] - 1, t_short.shape[0] - 1 - diff, -1, 
                             device=t_short.device) % t_short.shape[0]
        # Clamp indices to valid range (just in case)
        indices = torch.clamp(indices, 0, t_short.shape[0] - 1)
        # Get the padding elements
        padding = t_short[indices]
        # Concatenate with original tensor
        t_padded = torch.cat([t_short, padding], dim=0)
    else:
        t_padded = t_short

    return t_padded + t_long


class SeqToANNContainer(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x_seq: torch.Tensor):
        """
        Convert sequential input to ANN input and forward propagate
        :param x_seq: Input tensor, shape=[T, batch_size, ...]
        :return: Output tensor, shape=[T, batch_size, ...]
        """
        T, B = x_seq.shape[:2]
        y = x_seq.flatten(0, 1)  # [T*B, ...]

        for module in self:
            # if isinstance(module, nn.BatchNorm2d):
            #     y = y.view(T, B, *y.shape[1:])
            #     y = module(y.mean(dim=0))
            #     T = 1
            # else:
            #     y = module(y)

            y = module(y)

        return y.view(T, B, *y.shape[1:])


class CostCalculator:
    """A class to calculate and track FLOPs for a model with mixed timesteps."""
    
    def __init__(
            self, model: nn.Module, 
            input_size: Tuple[int, int, int, int], 
            target_T: float = 4.0,
            custom_hooks: Dict = {},
            custom_timesteps: Dict = {}
        ):
        """Initialize the FLOPs calculator.
        
        Args:
            model: The model to calculate FLOPs for
            input_size: Tuple of (batch_size, channels, height, width)
        """
        self.model = model
        self.input_size = input_size
        self.target_T = target_T
        self.custom_hooks = custom_hooks
        self.custom_timesteps = custom_timesteps

        # Track modules that should be skipped
        self.skip_modules: Set[nn.Module] = set()
        self._collect_skip_modules()
        
        self.flops_list = self._get_flops_list()
    
    def _collect_skip_modules(self):
        """Collect all submodules that should be skipped from custom modules."""
        for module in self.model.modules():
            if any(isinstance(module, t) for t in self.custom_hooks.keys()):
                # Add all submodules of custom modules to skip set
                for submodule in module.modules():
                    if submodule is not module:  # Skip the module itself
                        self.skip_modules.add(submodule)
    
    def _get_flops_list(self) -> torch.Tensor:
        """Calculate FLOPs list for the model using hooks."""
        flops_list = []
        current_flops = torch.DoubleTensor([0.])
        
        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            nonlocal current_flops
            if type(module) in register_hooks.keys():
                current_flops += register_hooks[type(module)](module, input, output)
            elif type(module) in self.custom_hooks.keys():
                flops_list.extend(self.custom_hooks[type(module)](module, input, output))
            elif isinstance(module, PseudoNeuron):
                flops_list.append(current_flops)
                current_flops = torch.DoubleTensor([0.])
        
        # Register hooks only for modules that should not be skipped
        handles = []
        for module in self.model.modules():
            # Skip modules in skip_modules set
            if module in self.skip_modules:
                continue
            
            # Skip Sequential and ModuleList
            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Run forward pass at training state
        training_state = self.model.training
        self.model.train()
        with torch.no_grad():
            # Create input tensor on the same device as model
            device = next(self.model.parameters()).device
            input = torch.randn(self.input_size, device=device).unsqueeze(0)
            self.model(input)
        self.model.train(training_state)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Add the last current_flops
        flops_list.append(current_flops)
        
        return torch.cat(flops_list) / 1e6
    
    def _get_timesteps_list(self) -> torch.Tensor:
        """Get timesteps list for all neurons in the model."""
        T_list = []
        
        for module in self.model.modules():
            if module in self.skip_modules:
                continue

            if type(module) in self.custom_timesteps:
                prev_T = T_list[-1] if T_list else torch.tensor(1., device=next(self.model.parameters()).device)
                T_list.extend(self.custom_timesteps[type(module)](module, prev_T=prev_T))

            if isinstance(module, PseudoNeuron):
                T_list.append(module.get_timesteps())

        
        return torch.stack(T_list)
    
    def calc_flops(self) -> torch.Tensor:
        """Calculate the current total FLOPs."""
        T_list = self._get_timesteps_list()
        flops_list = self.flops_list / self.flops_list.sum()
        # return (flops_list.to(T_list) * T_list).sum()
        return (flops_list.to(T_list)[1:] * T_list).sum()
    
    def calc_timesteps(self) -> torch.Tensor:
        """Calculate the current total timesteps."""
        T_list = self._get_timesteps_list()
        return T_list.mean()
    
    def cost(self) -> torch.Tensor:
        """Calculate the cost of the model."""
    
        energy_cost = self.calc_flops()
        time_cost = self.calc_timesteps()
        return energy_cost + time_cost

        # return self.calc_timesteps()

        # return self.calc_flops()
