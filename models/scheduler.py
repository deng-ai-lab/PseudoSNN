import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from neuron import PseudoNeuron


class MutiStepNoisyRateScheduler:

    def __init__(self, init_p=0.9, reduce_ratio=0.8, milestones=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], num_epoch=100, start_epoch=0):
        self.reduce_ratio = reduce_ratio
        self.p = init_p
        self.milestones = [int(m * num_epoch) for m in milestones]
        self.num_epoch = num_epoch
        self.start_epoch = start_epoch

    def set_noisy_rate(self, p, model):
        print(f'change noise rate as {p:.3f}')
        for m in model.modules():
            if isinstance(m, PseudoNeuron):
                m.noise_prob = p

    def __call__(self, epoch, model):
        if epoch <= self.start_epoch:
            self.set_noisy_rate(1. - self.p, model)

        if epoch - self.start_epoch in self.milestones:
            self.p *= self.reduce_ratio
            self.set_noisy_rate(1. - self.p, model)