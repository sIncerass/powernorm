#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskBatchNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['MaskBatchNorm']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class MaskBatchNorm(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.use_run = False
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, ' \
               'affine={affine}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        T, B, C = input.shape
        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None:
            mask_input = input.contiguous().view(-1, C)
        else:
            # Transpose the bn_mask (B x T -> T x B)
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)
            mask_input = input[bn_mask, :]

        input = input.permute(1, 2, 0).contiguous()
        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        mask_input = mask_input.unsqueeze(-1)
        sum_size = mask_input.size(0) * mask_input.size(2)
        input_sum = _sum_ft(mask_input)
        input_ssum = _sum_ft(mask_input ** 2)

        # Compute the statistics
        if self.training or not self.use_run:
            mean, inv_std = self._compute_mean_std(input_sum, input_ssum, sum_size)
        else:
            mean, inv_std = self.running_mean, self.running_var.clamp(self.eps) ** -0.5

        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        output = output.view(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        # Reshape it.
        if shaped_input:
            output = output.squeeze(0)
        return output

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5