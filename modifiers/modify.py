import inspect
from typing import List
from functools import partial

import torch.nn as nn

from .activations import ACTIVATION_NAMES_MAP, ActivationClass
from .normalizations import NORMALIZATION_NAMES_MAP, NormalizationClass


##########################################################################
#                Function to replace layers in a module                  #
##########################################################################

def create_layer_with_parameters(new_cls: nn.Module, old_layer: nn.Module, params_to_copy: List[str]) -> nn.Module:
    new_layer_args = inspect.signature(new_cls.__init__).parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in new_layer_args.values())
    layer_params = {
        k: v for k, v in old_layer.__dict__.items()
        if (
            k not in params_to_copy
            and not k.startswith('_')
            and (k in new_layer_args or has_kwargs)
        )
    }

    new_layer = new_cls(**layer_params)

    for p in params_to_copy:
        if hasattr(old_layer, p) and hasattr(new_layer, p):
            setattr(new_layer, p, getattr(old_layer, p))

    return new_layer

def replace_activation(
    module: nn.Module,
    original_activation: ActivationClass = 'GELU',
    replaced_activation: ActivationClass | nn.Module = 'ReLU',
    debug_info: bool = False,
) -> List[nn.Module]:
    assert original_activation in ACTIVATION_NAMES_MAP, f"Original activation '{original_activation}' is not supported."
    assert replaced_activation in ACTIVATION_NAMES_MAP or isinstance(replaced_activation, nn.Module), f"Replaced activation '{replaced_activation}' is not supported."

    original_cls = ACTIVATION_NAMES_MAP.get(original_activation)
    replaced_cls = ACTIVATION_NAMES_MAP.get(replaced_activation) if isinstance(replaced_activation, str) else replaced_activation

    resulting_layers: List[nn.Module] = []
    
    params_to_copy = ['training']
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if isinstance(child, original_cls):
                new_activation = create_layer_with_parameters(replaced_cls, child, params_to_copy=params_to_copy)

                if hasattr(new_activation, 'debug_info'):
                    new_activation.debug_info = debug_info

                setattr(layer, child_name, new_activation)
                resulting_layers.append(new_activation)

    return resulting_layers


def replace_normalization(
    module: nn.Module,
    original_normalization: NormalizationClass = 'BatchNorm2d',
    replaced_normalization: NormalizationClass | nn.Module = 'SparseBatchNorm2dQuantile50',
) -> List[nn.Module]:
    assert original_normalization in NORMALIZATION_NAMES_MAP, f"Original normalization '{original_normalization}' is not supported."
    assert replaced_normalization in NORMALIZATION_NAMES_MAP or isinstance(replaced_normalization, nn.Module), f"Replaced normalization '{replaced_normalization}' is not supported."

    original_cls = NORMALIZATION_NAMES_MAP.get(original_normalization)
    replaced_cls = NORMALIZATION_NAMES_MAP.get(replaced_normalization) if isinstance(replaced_normalization, str) else replaced_normalization

    resulting_layers: List[nn.Module] = []
    
    params_to_copy = ['training', 'running_mean', 'running_var']

    for layer in module.modules():
        for child_name, child in layer.named_children():
            if isinstance(child, original_cls):
                new_normalization = create_layer_with_parameters(replaced_cls, child, params_to_copy=params_to_copy)

                setattr(layer, child_name, new_normalization)
                resulting_layers.append(new_normalization)

    return resulting_layers


relufiaction = partial(replace_activation, replaced_activation='ReLU')
