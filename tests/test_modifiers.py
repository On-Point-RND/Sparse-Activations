
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifiers.normalizations import (
    QuantileBatchNorm2d,
)
from modifiers.activations import (
    TopKSparseGELU,
)
from modifiers.modify import (
    replace_activation,
    replace_normalization,
    relufiaction,
)

from tests.fixtures import *


# ============================================================================
# TESTS - Modifier functions
# ============================================================================

class TestReplaceActivation:
    """Tests for replace_activation function."""

    def test_replaces_gelu_with_relu(self, model_with_gelu):
        """Should replace GELU with ReLU."""
        layers = replace_activation(model_with_gelu, 'GELU', 'ReLU')

        assert isinstance(layers, list)
        assert len(layers) == 2
        assert all(isinstance(layer, nn.ReLU) for layer in layers)

        for layer in model_with_gelu.modules():
            assert not isinstance(layer, nn.GELU)

    def test_dont_replaces_silu_with_relu(self, model_with_gelu):
        """Should not replace SiLU with ReLU."""
        layers = replace_activation(model_with_gelu, 'SiLU', 'ReLU')

        assert len(layers) == 0

        for layer in model_with_gelu.modules():
            assert not isinstance(layer, nn.ReLU)

    def test_preserves_model_functionality(self, model_with_gelu):
        """Model should still be functional after replacement."""
        replace_activation(model_with_gelu, 'GELU', 'ReLU')

        x = torch.randn(2, 10)
        y = model_with_gelu(x)

        assert y.shape == (2, 10)

    def test_rejects_invalid_original_activation(self, model_with_gelu):
        """Should raise AssertionError for invalid original activation."""
        with pytest.raises(AssertionError):
            replace_activation(model_with_gelu, 'InvalidActivation', 'ReLU')

    def test_rejects_invalid_replaced_activation(self, model_with_gelu):
        """Should raise AssertionError for invalid replacement activation."""
        with pytest.raises(AssertionError):
            replace_activation(model_with_gelu, 'GELU', 'InvalidActivation')

    def test_preserves_class_parameters(self, model_with_gelu):
        """Should preserve parameters of replaced activations."""
        initial_approximate_values = [layer.approximate for layer in model_with_gelu if isinstance(layer, nn.GELU)]

        layers = replace_activation(model_with_gelu, 'GELU', 'TopKSparseGELU-10')

        assert len(layers) == 2
        assert all(isinstance(layer, TopKSparseGELU) for layer in layers)
        assert all(layer.approximate == approximate for layer, approximate in zip(layers, initial_approximate_values))


class TestReplaceNormalization:
    """Tests for replace_normalization function."""

    def test_replaces_batchnorm_with_quantile(self, model_with_batchnorm):
        """Should replace BatchNorm2d with QuantileBatchNorm2d."""
        layers = replace_normalization(
            model_with_batchnorm, 'BatchNorm2d', 'QuantileBatchNorm2d'
        )

        assert len(layers) == 2
        assert all(isinstance(layer, QuantileBatchNorm2d) for layer in layers)

    def test_infers_num_features_correctly(self, model_with_batchnorm):
        """Should preserve num_features during replacement."""
        params = []
        for layer in model_with_batchnorm.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.running_mean = torch.randn(layer.num_features)
                layer.running_var = torch.rand(layer.num_features) + 0.1  # Avoid zero variance
                params.append((layer.num_features, layer.running_mean.clone(), layer.running_var.clone()))

        layers = replace_normalization(
            model_with_batchnorm, 'BatchNorm2d', 'QuantileBatchNorm2d'
        )

        assert len(layers) == 2
        for layer, (num_features, running_mean, running_var) in zip(layers, params):
            assert isinstance(layer, QuantileBatchNorm2d)
            assert layer.num_features == num_features
            assert torch.allclose(layer.running_mean, running_mean)
            assert torch.allclose(layer.running_var, running_var)


class TestRelufiaction:
    """Tests for relufiaction helper function."""

    def test_replaces_with_relu_by_default(self, model_with_gelu):
        """Should replace activations with ReLU."""
        layers = relufiaction(model_with_gelu)

        assert len(layers) == 2
        assert all(isinstance(layer, nn.ReLU) for layer in layers)
