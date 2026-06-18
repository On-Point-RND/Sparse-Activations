
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifiers.normalizations import (
    BatchNorm2d,
    LayerNorm,
    QuantileBatchNorm2d,
    QuantileLayerNorm,
    QuantileMeanBatchNorm2d,
    BatchNorm2dPreStop,
    NORMALIZATION_NAMES_MAP,
)


from tests.fixtures import *

# ============================================================================
# TESTS - Normalization layers
# ============================================================================

class TestBatchNorm2d:
    """Tests for custom BatchNorm2d implementation."""

    def test_output_shape_matches_input(self, batch_norm_input):
        """Output shape should match input shape."""
        module = BatchNorm2d(32)
        original_module = nn.BatchNorm2d(32)

        y = module(batch_norm_input)
        expected_y = original_module(batch_norm_input)

        assert y.shape == batch_norm_input.shape
        assert torch.allclose(y, expected_y, atol=1e-5)

        assert torch.allclose(module.running_mean, original_module.running_mean, atol=1e-5)
        assert torch.allclose(module.running_var, original_module.running_var, atol=1e-3)

    def test_running_stats_tracked_in_training(self, batch_norm_input):
        """Running mean/var should be updated during training."""
        module = BatchNorm2d(32)
        module.train()

        # Get initial running mean
        initial_mean = module.running_mean.clone()

        # Forward pass
        module(batch_norm_input)

        # Running mean should change
        assert not torch.allclose(module.running_mean, initial_mean)

    def test_affine_parameters_are_trainable(self):
        """Weight and bias parameters should be trainable."""
        module = BatchNorm2d(32, affine=True)

        assert module.weight is not None
        assert module.bias is not None
        assert module.weight.requires_grad
        assert module.bias.requires_grad

    def test_no_affine_parameters_when_disabled(self):
        """When affine=False, weight and bias should be None."""
        module = BatchNorm2d(32, affine=False)

        assert module.weight is None
        assert module.bias is None


class TestLayerNorm:
    """Tests for custom LayerNorm implementation."""

    def test_output_shape_matches_input(self, layer_norm_input):
        """Output shape should match input shape."""
        module = LayerNorm(100)
        original_module = nn.LayerNorm(100)

        y = module(layer_norm_input)
        expected_y = original_module(layer_norm_input)

        assert y.shape == layer_norm_input.shape
        assert torch.allclose(y, expected_y, atol=1e-5)

    def test_normalized_mean_is_near_zero(self, layer_norm_input):
        """Normalized output mean should be close to 0."""
        module = LayerNorm(100)
        y = module(layer_norm_input)

        mean = y.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_normalized_std_is_near_one(self, layer_norm_input):
        """Normalized output std should be close to 1."""
        module = LayerNorm(100)
        y = module(layer_norm_input)

        std = y.std(dim=-1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)


class TestQuantileBatchNorm2d:
    """Tests for QuantileBatchNorm2d normalization."""

    def test_behaves_like_batchnorm_without_sparsity(self, batch_norm_input):
        """Without sparsity_level, should behave normally."""
        module = QuantileBatchNorm2d(32, sparsity_level=None)
        y = module(batch_norm_input)

        assert y.shape == batch_norm_input.shape

    @pytest.mark.parametrize('mode', ['global', 'batchwise', 'channelwise'])
    def test_different_quantile_search_modes(self, batch_norm_input, mode):
        """Should work with different search modes."""
        module = QuantileBatchNorm2d(
            32, sparsity_level=0.5, quantile_search_mode=mode
        )
        module.train()
        y = module(batch_norm_input)

        assert y.shape == batch_norm_input.shape

    def test_max_tracked_cnt_stops_updates(self, batch_norm_input):
        """Updates should stop after max_tracked_cnt is reached."""
        module = QuantileBatchNorm2d(32, sparsity_level=0.5, max_tracked_cnt=2)
        module.train()

        # Process first 2 batches
        module(batch_norm_input)
        module(batch_norm_input)
        mean_after_2 = module.running_mean.clone()

        # Process 3rd batch
        module(batch_norm_input)
        mean_after_3 = module.running_mean.clone()

        assert torch.allclose(mean_after_2, mean_after_3)


class TestQuantileLayerNorm:
    """Tests for QuantileLayerNorm normalization."""

    def test_behaves_like_layernorm_without_sparsity(self, layer_norm_input):
        """Without sparsity_level, should behave like LayerNorm."""
        module = QuantileLayerNorm(100, sparsity_level=None)
        y = module(layer_norm_input)

        assert y.shape == layer_norm_input.shape

    @pytest.mark.parametrize('invalid_sparsity', [0.0, 1.0])
    def test_rejects_invalid_sparsity_levels(self, invalid_sparsity):
        """Should reject sparsity_level=0.0 or 1.0."""
        with pytest.raises(AssertionError):
            QuantileLayerNorm(100, sparsity_level=invalid_sparsity)

    @pytest.mark.parametrize('mode', ['global', 'batchwise', 'channelwise'])
    def test_different_quantile_search_modes(self, layer_norm_input, mode):
        """Should work with different search modes."""
        module = QuantileLayerNorm(
            100, sparsity_level=0.5, quantile_search_mode=mode
        )
        module.train()
        y = module(layer_norm_input)

        assert y.shape == layer_norm_input.shape


class TestQuantileMeanBatchNorm2d:
    """Tests for QuantileMeanBatchNorm2d normalization."""

    def test_output_shape_matches_input(self, batch_norm_input):
        """Output shape should match input shape."""
        module = QuantileMeanBatchNorm2d(32, sparsity_level=0.5)
        module.train()
        y = module(batch_norm_input)

        assert y.shape == batch_norm_input.shape


class TestBatchNorm2dPreStop:
    """Tests for BatchNorm2dPreStop normalization."""

    def test_stops_updating_after_max_tracked_cnt(self, batch_norm_input):
        """Updates should stop after max_tracked_cnt is reached."""
        module = BatchNorm2dPreStop(32, max_tracked_cnt=2)
        module.train()

        # Process first 2 batches
        module(batch_norm_input)
        module(batch_norm_input)
        mean_after_2 = module.running_mean.clone()

        # Process 3rd batch
        module(batch_norm_input)
        mean_after_3 = module.running_mean.clone()

        assert torch.allclose(mean_after_2, mean_after_3)


# ============================================================================
# TESTS - Property preservation and edge cases
# ============================================================================


class TestNormalizationRegistry:
    """Tests for all normalizations in NORMALIZATION_NAMES_MAP."""

    @pytest.mark.parametrize('name', list(NORMALIZATION_NAMES_MAP.keys()))
    def test_all_normalizations_are_instantiable(self, name):
        """All registered normalizations should be instantiable."""
        cls_or_partial = NORMALIZATION_NAMES_MAP[name]

        if callable(cls_or_partial):
            if 'BatchNorm' in name or 'QuantileBatchNorm' in name:
                instance = cls_or_partial(3)
            else:
                instance = cls_or_partial(100)

            assert instance is not None

    @pytest.mark.parametrize('name', list(NORMALIZATION_NAMES_MAP.keys()))
    def test_all_normalizations_have_forward_pass(self, name):
        """All normalizations should support forward pass."""
        cls_or_partial = NORMALIZATION_NAMES_MAP[name]

        if callable(cls_or_partial):
            if 'BatchNorm' in name:
                instance = cls_or_partial(3)
                x = torch.randn(2, 3, 4, 4)
            else:
                instance = cls_or_partial(100)
                x = torch.randn(4, 100)

            y = instance(x)
            assert y.shape == x.shape
