import inspect
from typing import Dict, List
from functools import partial

import torch.nn as nn

from .activations import ACTIVATION_NAMES_MAP, ActivationClass
from .normalizations import NORMALIZATION_NAMES_MAP, NormalizationClass
from .linears import LINEAR_NAMES_MAP, LinearClass


##########################################################################
#                Function to replace layers in a module                  #
##########################################################################

def create_layer_with_parameters(new_cls: nn.Module, old_layer: nn.Module, params_to_copy: List[str] = [], params_to_set: Dict[str, object] = {}) -> nn.Module:
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

    for p, v in params_to_set.items():
        if p in new_layer_args or has_kwargs:
            layer_params[p] = v

    new_layer = new_cls(**layer_params)

    for p in params_to_copy:
        if (p not in layer_params) and hasattr(old_layer, p) and hasattr(new_layer, p):
            setattr(new_layer, p, getattr(old_layer, p))

    for p, v in params_to_set.items():
        if (p not in layer_params) and hasattr(new_layer, p):
            setattr(new_layer, p, v)

    return new_layer

def replace_activation(
    module: nn.Module,
    original_activation: ActivationClass = 'GELU',
    replaced_activation: ActivationClass | nn.Module = 'ReLU',
    **params_to_set,
) -> List[nn.Module]:
    assert original_activation in ACTIVATION_NAMES_MAP, f"Original activation '{original_activation}' is not supported."
    assert replaced_activation in ACTIVATION_NAMES_MAP or isinstance(replaced_activation, nn.Module), f"Replaced activation '{replaced_activation}' is not supported."

    original_cls = ACTIVATION_NAMES_MAP.get(original_activation)
    replaced_cls = ACTIVATION_NAMES_MAP.get(replaced_activation) if isinstance(replaced_activation, str) else replaced_activation

    resulting_layers: List[nn.Module] = []
    
    params_to_copy = ['training', 'num_batches_tracked', 'running_treshold']
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if type(child) is original_cls:
                new_activation = create_layer_with_parameters(replaced_cls, child, params_to_copy=params_to_copy, params_to_set=params_to_set)

                setattr(layer, child_name, new_activation)
                resulting_layers.append(new_activation)

    return resulting_layers


def replace_normalization(
    module: nn.Module,
    original_normalization: NormalizationClass = 'BatchNorm2d',
    replaced_normalization: NormalizationClass | nn.Module = 'QuantileBatchNorm2d-50',
    **params_to_set,
) -> List[nn.Module]:
    assert original_normalization in NORMALIZATION_NAMES_MAP, f"Original normalization '{original_normalization}' is not supported."
    assert replaced_normalization in NORMALIZATION_NAMES_MAP or isinstance(replaced_normalization, nn.Module), f"Replaced normalization '{replaced_normalization}' is not supported."

    original_cls = NORMALIZATION_NAMES_MAP.get(original_normalization)
    replaced_cls = NORMALIZATION_NAMES_MAP.get(replaced_normalization) if isinstance(replaced_normalization, str) else replaced_normalization

    resulting_layers: List[nn.Module] = []
    
    params_to_copy = ['training', 'running_mean', 'running_var']

    for layer in module.modules():
        for child_name, child in layer.named_children():
            if type(child) is original_cls:
                new_normalization = create_layer_with_parameters(replaced_cls, child, params_to_copy=params_to_copy, params_to_set=params_to_set)

                setattr(layer, child_name, new_normalization)
                resulting_layers.append(new_normalization)

    return resulting_layers


def replace_linear(
    module: nn.Module,
    original_linear: LinearClass = 'Linear',
    replaced_linear: LinearClass | nn.Module = 'TopKSparseLinear-50',
    **params_to_set,
) -> List[nn.Module]:
    assert original_linear in LINEAR_NAMES_MAP, f"Original linear '{original_linear}' is not supported."
    assert replaced_linear in LINEAR_NAMES_MAP or isinstance(replaced_linear, nn.Module), f"Replaced linear '{replaced_linear}' is not supported."

    original_cls = LINEAR_NAMES_MAP.get(original_linear)
    replaced_cls = LINEAR_NAMES_MAP.get(replaced_linear) if isinstance(replaced_linear, str) else replaced_linear

    resulting_layers: List[nn.Module] = []
    
    params_to_copy = ['training', 'transposed', 'output_padding', 'bias']
    
    for layer in module.modules():
        for child_name, child in layer.named_children():
            if type(child) is original_cls:
                new_activation = create_layer_with_parameters(replaced_cls, child, params_to_copy=params_to_copy, params_to_set=params_to_set)

                setattr(layer, child_name, new_activation)
                resulting_layers.append(new_activation)

    return resulting_layers


##########################################################################
#                Wrapper functions for common modifications              #
##########################################################################


relufiaction = partial(replace_activation, replaced_activation='ReLU')


def make_analytical_activation(module: nn.Module) -> List[nn.Module]:
    return sum(
        [
            replace_activation(module, original_activation=original_activation, replaced_activation=analytical_cls)
            for original_activation, analytical_cls in [
                ('GELU', 'AGELU'),
                ('ReLU', 'AReLU'),
                ('SiLU', 'ASiLU'),
            ]
        ], start=[]
    )


def make_analytical_linear(module: nn.Module) -> List[nn.Module]:
    return sum(
        [
            replace_linear(module, original_linear=original_linear, replaced_linear=analytical_cls)
            for original_linear, analytical_cls in [
                ('Linear', 'ALinear'),
                ('Conv2d', 'AConv2d'),
                ('Conv1d', 'AConv1d'),
            ]
        ], start=[]
    )
