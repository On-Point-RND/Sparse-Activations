from typing import Literal
from functools import partial

import torch.nn as nn

from .decorators import analytical_linear_module, topk_sparse_module


##########################################################################
#                           Sparse activations                           #
##########################################################################

@analytical_linear_module
@topk_sparse_module
class TopKSparseLinear(nn.Linear):
    """
    TopKSparseLinear is a variant of the linear layer that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


@analytical_linear_module
@topk_sparse_module
class TopKSparseConv2d(nn.Conv2d):
    """
    TopKSparseConv2d is a variant of the 2D convolutional layer that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


@analytical_linear_module
@topk_sparse_module
class TopKSparseConv1d(nn.Conv1d):
    """
    TopKSparseConv1d is a variant of the 1D convolutional layer that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


##########################################################################
#            Mapping from string names to linear classes                 #
##########################################################################

LINEAR_NAMES_MAP = {
    'Linear': nn.Linear,
    'Conv2d': nn.Conv2d,
    'Conv1d': nn.Conv1d,

    'ALinear': analytical_linear_module(nn.Linear),
    'AConv2d': analytical_linear_module(nn.Conv2d),
    'AConv1d': analytical_linear_module(nn.Conv1d),

    'TopKSparseLinear': TopKSparseLinear,
    'TopKSparseLinear-10': partial(TopKSparseLinear, sparsity_level=0.10),
    'TopKSparseLinear-25': partial(TopKSparseLinear, sparsity_level=0.25),
    'TopKSparseLinear-50': partial(TopKSparseLinear, sparsity_level=0.50),
    'TopKSparseLinear-75': partial(TopKSparseLinear, sparsity_level=0.75),
    'TopKSparseLinear-90': partial(TopKSparseLinear, sparsity_level=0.90),

    'TopKSparseConv2d': TopKSparseConv2d,
    'TopKSparseConv2d-10': partial(TopKSparseConv2d, sparsity_level=0.10),
    'TopKSparseConv2d-25': partial(TopKSparseConv2d, sparsity_level=0.25),
    'TopKSparseConv2d-50': partial(TopKSparseConv2d, sparsity_level=0.50),
    'TopKSparseConv2d-75': partial(TopKSparseConv2d, sparsity_level=0.75),
    'TopKSparseConv2d-90': partial(TopKSparseConv2d, sparsity_level=0.90),

    'TopKSparseConv1d': TopKSparseConv1d,
    'TopKSparseConv1d-10': partial(TopKSparseConv1d, sparsity_level=0.10),
    'TopKSparseConv1d-25': partial(TopKSparseConv1d, sparsity_level=0.25),
    'TopKSparseConv1d-50': partial(TopKSparseConv1d, sparsity_level=0.50),
    'TopKSparseConv1d-75': partial(TopKSparseConv1d, sparsity_level=0.75),
    'TopKSparseConv1d-90': partial(TopKSparseConv1d, sparsity_level=0.90),
}

LinearClass = Literal[
    'Linear', 'Conv2d', 'Conv1d',
    'ALinear', 'AConv2d', 'AConv1d',

    'TopKSparseLinear', 'TopKSparseLinear-10', 'TopKSparseLinear-25', 'TopKSparseLinear-50', 'TopKSparseLinear-75', 'TopKSparseLinear-90',
    'TopKSparseConv2d', 'TopKSparseConv2d-10', 'TopKSparseConv2d-25', 'TopKSparseConv2d-50', 'TopKSparseConv2d-75', 'TopKSparseConv2d-90',
    'TopKSparseConv1d', 'TopKSparseConv1d-10', 'TopKSparseConv1d-25', 'TopKSparseConv1d-50', 'TopKSparseConv1d-75', 'TopKSparseConv1d-90',
]
