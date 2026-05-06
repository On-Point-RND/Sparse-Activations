from typing import Optional, Literal
from functools import partial

import torch
import torch.nn as nn

from .utils import _review_as_with_batch


##########################################################################
#          Hand-written implementations of normalization layers          #
##########################################################################

class BatchNorm2d(nn.Module):
    """
    Hand-crafted implementation of 2D Batch Normalization.
    This module normalizes the input across the batch dimension

    for each channel independently.
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True
        ):
        """
        Args:
            num_features (int): Number of feature channels in the input.
            eps (float): A small value to avoid division by zero.
            momentum (float): Momentum for running mean and variance.
            affine (bool): If True, learnable scale and shift parameters are used.
            track_running_stats (bool): If True, running mean and variance are tracked.
        """
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
            self.running_mean: torch.Tensor | None
            self.running_var: torch.Tensor | None
            self.num_batches_tracked: torch.Tensor | None
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(
            self,
            x: torch.Tensor,
            batch_mean: Optional[torch.Tensor] = None,
            batch_var: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Forward pass for batch normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            batch_mean (Optional[torch.Tensor]): Optional precomputed batch mean.
            batch_var (Optional[torch.Tensor]): Optional precomputed batch variance.
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        if x.dim() != 4:
            raise ValueError("Expected input tensor to be 4D (N, C, H, W)")
        N, C, H, W = x.shape
        if C != self.num_features:
            raise ValueError(f"Expected input with {self.num_features} channels, got {C}")
        
        if self.training or (self.running_mean is None) or (self.running_var is None):
            if batch_mean is None:
                batch_mean = x.mean(dim=(0, 2, 3))
            if batch_var is None:
                batch_var = (x - _review_as_with_batch(batch_mean, x.shape)).square().mean(dim=(0, 2, 3))

            if self.track_running_stats:
                if self.momentum is None or self.momentum == 1.0:
                    self.running_mean = batch_mean
                    self.running_var = batch_var
                else:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_normalized = (x - _review_as_with_batch(batch_mean, x.shape)) / (_review_as_with_batch(batch_var, x.shape) + self.eps).sqrt()

        if self.affine:
            x_normalized = x_normalized * _review_as_with_batch(self.weight, x.shape) + _review_as_with_batch(self.bias, x.shape)

        return x_normalized
    
    def extra_repr(self) -> str:
        return (f'num_features={self.num_features}, eps={self.eps}, '
                f'momentum={self.momentum}, affine={self.affine}, '
                f'track_running_stats={self.track_running_stats}')
    

class LayerNorm(nn.Module):
    """
    Hand-crafted implementation of Layer Normalization.
    """
    def __init__(
            self,
            normalized_shape,
            eps: float = 0.00001,
            elementwise_affine: bool = True,
            bias: bool = True
        ):
        """
        Args:
            normalized_shape (int or list or torch.Size): Input shape from an expected input.
            eps (float): A small value to avoid division by zero.
            elementwise_affine (bool): If True, learnable scale and shift parameters are used.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(LayerNorm, self).__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(
            self,
            x: torch.Tensor,
            layer_mean: Optional[torch.Tensor] = None,
            layer_var: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor.
            layer_mean (Optional[torch.Tensor]): Optional precomputed layer mean.
            layer_var (Optional[torch.Tensor]): Optional precomputed layer variance.
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        
        # Determine the dimensions to normalize over (last len(normalized_shape) dims)
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        if layer_mean is None:
            layer_mean = x.mean(dim=dims, keepdim=True)
        if layer_var is None:
            layer_var = (x - layer_mean).square().mean(dim=dims, keepdim=True)

        x_normalized = (x - layer_mean) / (layer_var + self.eps).sqrt()

        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
            if self.bias is not None:
                x_normalized = x_normalized + self.bias

        return x_normalized
    
    def extra_repr(self) -> str:
        return (f'normalized_shape={self.normalized_shape}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}, bias={self.bias is not None}')
    

##########################################################################
#                     PreStop Normalization layers                       #
##########################################################################
    
    
class BatchNorm2dPreStop(BatchNorm2d):
    """
    BatchNorm2d variant that stops updating running statistics after a certain number of batches.
    After num_batches_tracked reaches max_tracked_cnt, the module will use the current running_mean and running_var for normalization without updating them further.
    """

    def __init__(self, *args, max_tracked_cnt: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_tracked_cnt = max_tracked_cnt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean = None
        batch_var = None

        if self.track_running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            batch_mean = self.running_mean
            batch_var = self.running_var

        return super().forward(x, batch_mean=batch_mean, batch_var=batch_var)
    
    def extra_repr(self) -> str:
        return f'(pre-stop) max_tracked_cnt={self.max_tracked_cnt}, {super().extra_repr()}'
    

class LayerNormPreStop(LayerNorm):
    """
    LayerNorm variant that stops updating running statistics after a certain number of batches.
    After num_batches_tracked reaches max_tracked_cnt, the module will use the current running_layer_mean for normalization without updating it further.
    """

    def __init__(
            self,
            *args,
            track_running_stats: bool = True,
            running_shape: Optional[torch.Size] = None,
            momentum: float = 0.1,
            max_tracked_cnt: Optional[int] = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.max_tracked_cnt = max_tracked_cnt

        if self.track_running_stats:
            self.register_buffer('running_layer_mean', torch.zeros(running_shape))
            self.register_buffer('running_layer_var', torch.ones(running_shape))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
            self.running_layer_mean: torch.Tensor | None
            self.running_layer_var: torch.Tensor | None
            self.num_batches_tracked: torch.Tensor | None
        else:
            self.register_buffer('running_layer_mean', None)
            self.register_buffer('running_layer_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_mean = None
        layer_var = None

        if self.track_running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            layer_mean = self.running_layer_mean
            layer_var = self.running_layer_var
        elif self.training or self.running_layer_mean is None:
            dims = tuple(range(-len(self.normalized_shape), 0))
            
            layer_mean = x.mean(dim=dims, keepdim=True)
            layer_var = (x - layer_mean).square().mean(dim=dims, keepdim=True)

            if self.track_running_stats:
                with torch.no_grad():
                    if self.num_batches_tracked != 0:
                        self.running_layer_mean = (1 - self.momentum) * self.running_layer_mean + self.momentum * layer_mean
                        self.running_layer_var = (1 - self.momentum) * self.running_layer_var + self.momentum * layer_var
                    else:
                        self.running_layer_mean = layer_mean
                        self.running_layer_var = layer_var
                    self.num_batches_tracked += 1
        else:
            layer_mean = self.running_layer_mean
            layer_var = self.running_layer_var

        output= super().forward(x, layer_mean=layer_mean, layer_var=layer_var)
        return output

    def extra_repr(self) -> str:
        return f'(pre-stop) max_tracked_cnt={self.max_tracked_cnt}, {super().extra_repr()}'


##########################################################################
#                 Quantile-based normalization layers                    #
##########################################################################

class QuantileBatchNorm2d(BatchNorm2d):
    def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
            max_tracked_cnt: Optional[int] = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.sparsity_level = sparsity_level
        self.quantile_search_mode = quantile_search_mode
        self.max_tracked_cnt = max_tracked_cnt

        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantile_search_mode]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean = None

        if self.sparsity_level is None:
            return super().forward(x)
        
        if self.track_running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            batch_mean = self.running_mean
        elif self.training and self.sparsity_level is not None:
            x_viewed = self.quantile_view_fn(x)

            kth_element = int(self.sparsity_level * x_viewed.size(dim=-1)) + 1
            batch_mean = torch.kthvalue(x_viewed, kth_element, dim=-1).values
            batch_mean = batch_mean.mean(dim=0) # Average over batch

        return super().forward(x, batch_mean=batch_mean)

    def extra_repr(self) -> str:
        return f'quantile={self.sparsity_level}, {super().extra_repr()}'
    

class QuantileMeanBatchNorm2d(BatchNorm2d):
    def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
            max_tracked_cnt: Optional[int] = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.sparsity_level = sparsity_level
        self.quantile_search_mode = quantile_search_mode
        self.max_tracked_cnt = max_tracked_cnt

        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantile_search_mode]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean, batch_var = None, None

        if self.sparsity_level is None:
            return super().forward(x)
        
        if self.track_running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            batch_mean = self.running_mean

            batch_var = x.var(dim=(0, 2, 3), correction=0)
        elif self.training and self.sparsity_level is not None:
            x_viewed = self.quantile_view_fn(x)

            kth_element = int(self.sparsity_level * x_viewed.size(dim=-1)) + 1
            batch_mean = torch.kthvalue(x_viewed, kth_element, dim=-1).values
            batch_mean = batch_mean.mean(dim=0) # Average over batch

            batch_var = x.var(dim=(0, 2, 3), correction=0)

        return super().forward(x, batch_mean=batch_mean, batch_var=batch_var)

    def extra_repr(self) -> str:
        return f'(standart var) quantile={self.sparsity_level}, {super().extra_repr()}'


class QuantileLayerNorm(LayerNorm):
    """
    Quantile version of the LayerNorm module.
    This module normalizes only the non-zero elements in the input tensor.
    """
    def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
            
            track_running_stats: bool = False,
            running_shape: Optional[torch.Size] = None,
            momentum: float = 0.1,
            max_tracked_cnt: Optional[int] = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        assert sparsity_level is None or (0.0 < sparsity_level < 1.0), \
            "sparsity_level must be in the range (0.0, 1.0)"

        self.sparsity_level = sparsity_level
        self.quantile_search_mode = quantile_search_mode
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.max_tracked_cnt = max_tracked_cnt

        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantile_search_mode]

        if self.track_running_stats:
            self.register_buffer('running_layer_mean', torch.ones(running_shape))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
            self.running_layer_mean: torch.Tensor | None
            self.num_batches_tracked: torch.Tensor | None
        else:
            self.register_buffer('running_layer_mean', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse batch normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        layer_mean = None

        if self.sparsity_level is None:
            return super().forward(x)

        if self.track_running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            layer_mean = self.running_layer_mean
        elif self.training or self.running_layer_mean is None:
            x_viewed = self.quantile_view_fn(x)

            # Compute quantile threshold
            kth_element = int(self.sparsity_level * x_viewed.size(dim=-1)) + 1
            layer_mean = torch.kthvalue(x_viewed, kth_element, dim=-1).values
            layer_mean = layer_mean.mean(dim=0) # Average over batch

            if self.track_running_stats:
                with torch.no_grad():
                    if self.num_batches_tracked != 0:
                        self.running_layer_mean = (1 - self.momentum) * self.running_layer_mean + self.momentum * layer_mean
                    else:
                        self.running_layer_mean = layer_mean
                    self.num_batches_tracked += 1
        else:
            layer_mean = self.running_layer_mean

        layer_mean = _review_as_with_batch(layer_mean, x.shape)
            
        return super().forward(x, layer_mean=layer_mean)
    
    def extra_repr(self):
        return f'quantile={self.sparsity_level}, {super().extra_repr()}'

##########################################################################
#         Mapping from string names to nomalization classes              #
##########################################################################

NORMALIZATION_NAMES_MAP = {
    'BatchNorm2d': nn.BatchNorm2d,
    'LayerNorm': nn.LayerNorm,

    'BatchNorm2dPreStop': BatchNorm2dPreStop,
    'LayerNormPreStop': LayerNormPreStop,

    'QuantileBatchNorm2d': QuantileBatchNorm2d,
    'QuantileBatchNorm2d-10': partial(QuantileBatchNorm2d, sparsity_level=0.1),
    'QuantileBatchNorm2d-25': partial(QuantileBatchNorm2d, sparsity_level=0.25),
    'QuantileBatchNorm2d-50': partial(QuantileBatchNorm2d, sparsity_level=0.50),
    'QuantileBatchNorm2d-75': partial(QuantileBatchNorm2d, sparsity_level=0.75),
    'QuantileBatchNorm2d-90': partial(QuantileBatchNorm2d, sparsity_level=0.90),

    'QuantileBatchNorm2d-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000),
    'QuantileBatchNorm2d-10-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000, sparsity_level=0.1),
    'QuantileBatchNorm2d-25-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000, sparsity_level=0.25),
    'QuantileBatchNorm2d-50-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000, sparsity_level=0.50),
    'QuantileBatchNorm2d-75-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000, sparsity_level=0.75),
    'QuantileBatchNorm2d-90-AS': partial(QuantileBatchNorm2d, max_tracked_cnt=50_000, sparsity_level=0.90),

    'QuantileMeanBatchNorm2d': QuantileMeanBatchNorm2d,
    'QuantileMeanBatchNorm2d-10': partial(QuantileMeanBatchNorm2d, sparsity_level=0.1),
    'QuantileMeanBatchNorm2d-25': partial(QuantileMeanBatchNorm2d, sparsity_level=0.25),
    'QuantileMeanBatchNorm2d-50': partial(QuantileMeanBatchNorm2d, sparsity_level=0.50),
    'QuantileMeanBatchNorm2d-75': partial(QuantileMeanBatchNorm2d, sparsity_level=0.75),
    'QuantileMeanBatchNorm2d-90': partial(QuantileMeanBatchNorm2d, sparsity_level=0.90),

    'QuantileLayerNorm': QuantileLayerNorm,
    'QuantileLayerNorm-10': partial(QuantileLayerNorm, sparsity_level=0.1),
    'QuantileLayerNorm-25': partial(QuantileLayerNorm, sparsity_level=0.25),
    'QuantileLayerNorm-50': partial(QuantileLayerNorm, sparsity_level=0.50),
    'QuantileLayerNorm-75': partial(QuantileLayerNorm, sparsity_level=0.75),
    'QuantileLayerNorm-90': partial(QuantileLayerNorm, sparsity_level=0.90),

    'QuantileLayerNorm-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000),
    'QuantileLayerNorm-10-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000, sparsity_level=0.1),
    'QuantileLayerNorm-25-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000, sparsity_level=0.25),
    'QuantileLayerNorm-50-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000, sparsity_level=0.50),
    'QuantileLayerNorm-75-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000, sparsity_level=0.75),
    'QuantileLayerNorm-90-AS': partial(QuantileLayerNorm, max_tracked_cnt=50_000, sparsity_level=0.90),
}

NormalizationClass = Literal[
    'BatchNorm2d',
    'LayerNorm',

    'BatchNorm2dPreStop',
    'LayerNormPreStop',
    
    'QuantileBatchNorm2d', 'QuantileBatchNorm2d-10', 'QuantileBatchNorm2d-25', 'QuantileBatchNorm2d-50', 'QuantileBatchNorm2d-75', 'QuantileBatchNorm2d-90',
    'QuantileBatchNorm2d-AS', 'QuantileBatchNorm2d-10-AS', 'QuantileBatchNorm2d-25-AS', 'QuantileBatchNorm2d-50-AS', 'QuantileBatchNorm2d-75-AS', 'QuantileBatchNorm2d-90-AS',

    'QuantileMeanBatchNorm2d', 'QuantileMeanBatchNorm2d-10', 'QuantileMeanBatchNorm2d-25', 'QuantileMeanBatchNorm2d-50', 'QuantileMeanBatchNorm2d-75', 'QuantileMeanBatchNorm2d-90',

    'QuantileLayerNorm', 'QuantileLayerNorm-10', 'QuantileLayerNorm-25', 'QuantileLayerNorm-50', 'QuantileLayerNorm-75', 'QuantileLayerNorm-90',
    'QuantileLayerNorm-AS', 'QuantileLayerNorm-10-AS', 'QuantileLayerNorm-25-AS', 'QuantileLayerNorm-50-AS', 'QuantileLayerNorm-75-AS', 'QuantileLayerNorm-90-AS',
]
