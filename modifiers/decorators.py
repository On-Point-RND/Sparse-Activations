from typing import Literal, Optional, Type

import torch
import torch.nn as nn

from .utils import _review_as_with_batch


##########################################################################
#                    Decorator for analizing modules                     #
##########################################################################

def analytical_activation_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Decorator to create an analytical version of a given nn.Module class. The resulting class will have additional attributes to store the input and output activations, as well as a debug_info flag to control whether these activations are stored during the forward pass.
    """

    class AnalyticalModule(cls):
        def __init__(
            self,
            *args,
            debug_info: bool = False,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.debug_info = debug_info
            self.in_activation = None
            self.out_activation = None

            if self.debug_info:
                def forward_hook(module, input, output):
                    self.in_activation = input[0].clone().detach()
                    self.out_activation = output.clone().detach()

                self.register_forward_hook(forward_hook)
        
        def extra_repr(self) -> str:
            return f'debug_info={self.debug_info}, {super().extra_repr()}'
        
    AnalyticalModule.__name__ = f"Analytical{cls.__name__}"
        
    return AnalyticalModule


def analytical_linear_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Decorator to create an analytical version of a given nn.Module class. The resulting class will have additional attributes to store the input and output activations, as well as a debug_info flag to control whether these activations are stored during the forward pass.
    """

    class AnalyticalModule(cls):
        def __init__(
            self,
            *args,
            debug_info: bool = False,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.debug_info = debug_info

            self.in_activation = None
            self.out_activation = None

            self.grad_in_activation = None
            self.grad_out_activation = None

            self.initial_weight_copy = None

            if self.debug_info:
                def forward_hook(module, input, output):
                    self.in_activation = input[0].clone().detach()
                    self.out_activation = output.clone().detach()

                def backward_hook(module, grad_input, grad_output):
                    if grad_input[0] is not None:
                        self.grad_in_activation = grad_input[0].clone().detach()
                    if grad_output[0] is not None:
                        self.grad_out_activation = grad_output[0].clone().detach()

                self.register_forward_hook(forward_hook)
                self.register_full_backward_hook(backward_hook)
        
        def extra_repr(self) -> str:
            return f'debug_info={self.debug_info}, {super().extra_repr()}'
        
        def record_initial_weights(self):
            self.initial_weight_copy = self.weight.detach().clone()
        
        @property
        def z_score(self):
            if self.initial_weight_copy is None:
                raise ValueError("Initial weights not recorded. Call record_initial_weights() before accessing z_score.")
            return (self.weight - self.initial_weight_copy).mean() / (self.initial_weight_copy.std() + self.weight.std() + 1e-8) * 2.0
        
        @property
        def weight_grad_mean(self):
            return self.weight.grad.abs().mean()
        
        @property
        def weight_grad_norm(self):
            return self.weight.grad.norm()
        
        @property
        def weight_mean(self):
            return self.weight.mean()
        
        @property
        def weight_norm(self):
            return self.weight.norm()
        
    AnalyticalModule.__name__ = f"Analytical{cls.__name__}"
        
    return AnalyticalModule


##########################################################################
#                    Decorator for sparse activations                    #
##########################################################################

def topk_sparse_module(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Decorator to create a sparse version of a given nn.Module class. The resulting class will have an additional attribute sparsity_level to control the level of sparsity applied to the activations during the forward pass. The sparsity is applied by zeroing out the smallest activations based on the specified sparsity level.
    """

    class SparseModule(cls):
        def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            post_sparsity: bool = True,
            quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',

            running_stats: bool = False,
            running_shape: Optional[torch.Size] = None,
            momentum: float = 0.1,
            max_tracked_cnt: Optional[int] = None,
            **kwargs
        ):
            super().__init__(*args, **kwargs)

            assert sparsity_level is None or (0.0 < sparsity_level < 1.0), "sparsity_level must be in (0, 1)"

            self.sparsity_level = sparsity_level
            self.post_sparsity = post_sparsity
            self.running_stats = running_stats
            self.momentum = momentum

            self.quantile_search_mode = quantile_search_mode
            self.quantile_view_fn = {
                'global': lambda x: x.view(-1),
                'batchwise': lambda x: x.view(x.size(0), -1),
                'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
            }[self.quantile_search_mode]

            if self.running_stats:
                self.running_treshold = torch.zeros(running_shape)
                self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
            else:
                self.register_buffer('running_treshold', None)
                self.register_buffer('num_batches_tracked', None)
                
            self.max_tracked_cnt = max_tracked_cnt

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.post_sparsity:
                x = super().forward(x)

            if self.sparsity_level is not None:
                if self.running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
                    treshold = self.running_treshold
                elif self.training or self.running_treshold is None:
                    # Compute quantile threshold
                    x_viewed = self.quantile_view_fn(x)
                    total_elements = x_viewed.size(dim=-1)  # per-sample element count
                    n_remove = int(self.sparsity_level * total_elements) + 1

                    treshold = torch.kthvalue(x_viewed, n_remove, dim=-1).values
                    treshold = treshold.mean(dim=0) # Average over batch

                    if self.running_stats:
                        with torch.no_grad():
                            if self.num_batches_tracked != 0:
                                self.running_treshold = (1 - self.momentum) * self.running_treshold + self.momentum * treshold
                            else:
                                self.running_treshold = treshold
                            self.num_batches_tracked += 1
                # Compute quantile threshold
                else:
                    treshold = self.running_treshold
                
                treshold = _review_as_with_batch(treshold, x.shape)
                mask = x < treshold
                x.masked_fill_(mask, 0.0)

            if not self.post_sparsity:
                x = super().forward(x)

            return x
        
        def extra_repr(self) -> str:
            return f'sparsity_level={self.sparsity_level}, {super().extra_repr()}'
        
    SparseModule.__name__ = f"TopK{cls.__name__}"

    return SparseModule
