import torch
import torch.nn as nn
from torch.nn import functional as F

from spikingjelly.activation_based import surrogate
from spikingjelly.clock_driven.neuron import IFNode


def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


class CustomIFNode(IFNode):
    def __init__(self, **kwargs):
        super(CustomIFNode, self).__init__(
            v_threshold=1.0,
            v_reset=None,
            surrogate_function=surrogate.Sigmoid(),
            **kwargs
        )

    def neuronal_fire(self):
        return self.surrogate_function(self.v + 0.5 - self.v_threshold)


class PseudoNeuron(nn.Module):
    def __init__(
            self, 
            noise_type: str = 'uniform',
            noise_prob: float = 0.5,
            init_T: float = 8.0,
            min_T: int = 1,
            max_T: int = 16,
            scale: float = 1.0,
        ):
        super(PseudoNeuron, self).__init__()

        self.noise_prob = noise_prob
        self.init_T = init_T
        self.min_T = min_T
        self.max_T = max_T

        self.spiking_function = CustomIFNode()
        # self.v = 0.0

        ratio = (self.init_T - self.min_T) / (self.max_T - self.min_T)
        assert 0 < ratio < 1
        self.logit = torch.nn.Parameter(torch.logit(torch.tensor(float(ratio))))

        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

        if noise_type == 'uniform':
            self.base_noise = lambda x: torch.rand_like(x) - 0.5
        elif noise_type == 'gaussian':
            self.base_noise = lambda x: torch.randn_like(x) / 3.
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")
        
            
    def get_timesteps(self):
        # STE ver.
        T = self.min_T + (self.max_T - self.min_T) * torch.sigmoid(self.logit)
        return torch.round(T) + (T - T.detach())
    
    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * self.noise_prob)
    
    def _apply_quantization_noise(self, x: torch.Tensor, T: torch.Tensor):
        """Apply quantization noise to the input"""
        # Original layer-wise implementation:
        if torch.rand(1) < self.noise_prob:
            mask = torch.logical_and(x > 0, x < 1.)
            x = x + self.base_noise(x) * mask * (1. / T)
        
        # # Element-wise quantization noise using self.mask
        # noise = self.base_noise(x) * (1. / T)
        # x = x + noise * self.create_mask(x)
        return x
    
    def _apply_clipping_noise(self, x: torch.Tensor):
        """Apply clipping noise to the input"""
        # Original layer-wise implementation:
        if torch.rand(1) < self.noise_prob:
            x = x.clamp(max=1.)
            # x -= F.relu(x - 1.) # TODO: check this
        
        # # Element-wise clipping noise using self.mask
        # # Apply clipping only where mask is 1, keep original value where mask is 0
        # clipped = x.clamp(max=1.)
        # x = torch.where(self.create_mask(x).bool(), clipped, x)
        return x
    
    def _inference_forward(self, x_seq: torch.Tensor, T: torch.Tensor):
        """Process timesteps during inference"""
        if x_seq.shape[0] == T:
            # Use each timestep individually
            input_seq = x_seq
        else:
            # Use mean across timesteps
            x_mean = x_seq.mean(dim=0)
            input_seq = x_mean.unsqueeze(0).expand(T.int(), *x_mean.shape)
        
        spike_seq = []
        for i in range(T.int()):
            spike_seq.append(self.spiking_function(input_seq[i]))
        
        return torch.stack(spike_seq, dim=0)
    
    def forward(self, x_seq: torch.Tensor):
        T = self.get_timesteps()
        x_seq = x_seq * self.scale + self.bias

        if self.training:
            x_seq = x_seq.mean(dim=0, keepdim=True)
            
            # Training mode: apply noise and update membrane potential
            # x_seq = x_seq + self.v
            
            # Apply noises
            x_seq = self._apply_quantization_noise(x_seq, T)
            x_seq = self._apply_clipping_noise(x_seq)
            
            # Apply activation and update membrane potential
            out = F.relu(x_seq)
            # self.v = x_seq - out
            
            return out
        else:
            # Inference mode: use spiking neuron
            return self._inference_forward(x_seq, T)

    # def reset(self):
    #     self.v = 0.0
        