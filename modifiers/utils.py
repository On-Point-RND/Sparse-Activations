import torch


##########################################################################
#             Auxiliary functions for normalization layers               #
##########################################################################

def _review_as_with_batch(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    extra_dims = len(target_shape) - len(x.shape) - 1
    return x.view(1, *x.shape, *((1,) * extra_dims))
