#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : groupnorm.py
# Distributed under MIT License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def tile(a, repeats, dim):
    """
    Substitute for numpy's repeat function. Taken from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)

class GroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_feature = num_channels // num_groups
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, pad_mask=None, is_encoder=False):
        # input: only reudce over the C dim.
        shaped_input = (len(input.shape) == 2)
        if shaped_input:
            input = input.unsqueeze(0)
        T, B, C = input.shape
        # Permute the mask_input to (B, T, C)
        # mask_input = input.transpose(0, 1)
        # # Compute the mean, var for LN, size to be BxTx1 -> BxCxT
        # # Split the mask_input into group
        # gn_input = mask_input.view(B, T, self.num_groups, self.group_feature)
        # gn_input = gn_input.permute(1, 2, 3, 0).contiguous().view(T, self.num_groups, self.group_feature * B)
        # # TxGx1 -> TxC -> BxTxC -> BxCxT
        # mean_gn = tile(gn_input.mean(-1, keepdim=True).squeeze(-1), self.group_feature, -1).expand_as(mask_input).transpose(1, 2)
        # var_gn = tile(gn_input.var(-1, keepdim=True).squeeze(-1), self.group_feature, -1).expand_as(mask_input).transpose(1, 2)
        #
        # # Resize the input to (B, C, -1).
        # input = input.permute(1, 2, 0).contiguous()
        # input_shape = input.size()
        # input = input.view(input.size(0), self.num_channels, -1)
        #
        # input = (input - mean_gn) / (var_gn + self.eps).sqrt()
        # input = input * (self.weight).unsqueeze(-1) + (self.bias).unsqueeze(-1)
        # input = input.view(B, C, T)
        # input = input.permute(2, 0, 1).contiguous()
        # return input

        input = input.contiguous().view(T*B, C)
        input = F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = input.contiguous().view(T, B, C)
        if shaped_input:
            input = input.squeeze(0)
        return input

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)