#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : MaskBatchNorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

__all__ = ['MaskSyncBatchNorm']

class LayerScaling1D(nn.Module):
    r"""Scales inputs by the second moment for the entire layer.
    .. math::
        y = \frac{x}{\sqrt{\mathrm{E}[x^2] + \epsilon}}
    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)` (same shape as input)
    Examples::
        >>> ls = LayerScaling()
        >>> input = torch.randn(20, 100)
        >>> output = ls(input)
    """
    def __init__(self, eps=1e-5, **kwargs):
        super(LayerScaling1D, self).__init__()
        self.eps = eps

    def extra_repr(self):
        return f'eps={self.eps}'

    def forward(self, input):
        # calculate second moment
        moment2 = torch.mean(input * input, dim=2, keepdim=True)
        # divide out second moment
        return input / torch.sqrt(moment2 + self.eps)

def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class MaskSyncBatchNorm(nn.Module):
    """
    An implementation of masked batch normalization, used for testing the numerical
    stability.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.05, \
        affine=True, track_running_stats=True, sync_bn=True, process_group=None):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.sync_bn = sync_bn
        # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.process_group = process_group
        self.ddp_gpu_size = 4
        self.lp = LayerScaling1D()
        self.reset_parameters()

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def forward(self, input, pad_mask=None, is_encoder=False, update_run=True):
        """
        input:  T x B x C -> B x C x T
             :  B x C x T -> T x B x C
        pad_mask: B x T (padding is True)
        """
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        input = self.lp( input )

        T, B, C = input.shape
        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None:
            mask_input = input.contiguous().view(-1, C)

        else:
            # Transpose the bn_mask (B x T -> T x B)
            bn_mask = ~pad_mask
            bn_mask = bn_mask.transpose(0, 1)
            # mask_input = input[bn_mask, :]
        if pad_mask is not None:
            pad_size = (~bn_mask).sum()
            mask_input = input.clone()
            mask_input[~bn_mask, :] = input[bn_mask, :].mean(dim=0).clone().unsqueeze(0).repeat( pad_size, 1 )
            mask_input = mask_input.contiguous().view(T*B, C)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        if not update_run:
            exponential_average_factor = 0.0

        need_sync = False #self.training #or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # print( need_sync, exponential_average_factor )
        # if fuse_dropout != 0.0:
        #     # estimate the real mean/variance for training:
        #     with torch.no_grad():
        #         if not need_sync:
        #             z = F.batch_norm(
        #                 mask_input, self.running_mean, self.running_var, self.weight, self.bias,
        #                 self.training or not self.track_running_stats,
        #                 exponential_average_factor, self.eps)
        #         else:
        #             # if not self.ddp_gpu_size:
        #             #     raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
        #             z = sync_batch_norm.apply(
        #                 mask_input, self.weight, self.bias, self.running_mean, self.running_var,
        #                 self.eps, exponential_average_factor, process_group, world_size)
        #     exponential_average_factor = 0.0
        #     input = F.dropout( input, p=fuse_dropout, training=self.training )
        #     if pad_mask is None:
        #         mask_input = input.contiguous().view(-1, C)
        #     else:
        #         pad_size = (~bn_mask).sum()
        #         mask_input = input.clone()
        #         mask_input[~bn_mask, :] = input[bn_mask, :].mean(dim=0).clone().unsqueeze(0).repeat( pad_size, 1 )
        #         mask_input = mask_input.contiguous().view(T*B, C)
        # import copy
        # shit_run_mean = copy.deepcopy(self.running_mean)
        # shit_run_var = copy.deepcopy(self.running_var)
        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            z = F.batch_norm(
                mask_input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            # if not self.ddp_gpu_size:
            #     raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
            z = sync_batch_norm.apply(
                mask_input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size)

        output = z
        # if exponential_average_factor == 0.0:
        # print( exponential_average_factor, "mean_diff_update", (self.running_mean - shit_run_mean).sum(), \
                # "var_diff_update", (self.running_var - shit_run_var).sum())
        # if pad_mask is None:
        #     output = z.clone()
        # else:
        #     output = input.clone()
        #     output[bn_mask, :] = z.clone()
        #     output[~bn_mask, :] = z.clone().mean(0)
        # input = input.permute(1, 2, 0).contiguous()
        # # Resize the input to (B, C, -1).
        # input_shape = input.size()
        # input = input.view(input.size(0), self.num_features, -1)

        # # Compute the sum and square-sum.
        # mask_input = mask_input.unsqueeze(-1)
        # sum_size = mask_input.size(0) * mask_input.size(2)
        # input_sum = _sum_ft(mask_input)
        # input_ssum = _sum_ft(mask_input ** 2)

        # # Compute the statistics
        # if self.training or not self.use_run:
        #     mean, inv_std = self._compute_mean_std(input_sum, input_ssum, sum_size)
        # else:
        #     mean, inv_std = self.running_mean, self.running_var.clamp(self.eps) ** -0.5

        # if self.affine:
        #     output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        # else:
        #     output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # output = output.view(input_shape)
        # output = output.permute(2, 0, 1).contiguous()
        output = output.view( T, B, C )
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