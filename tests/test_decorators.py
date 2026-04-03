

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifiers.decorators import analytical_module, topk_sparse_module

from tests.fixtures import *


activation_parametrization = pytest.mark.parametrize('activation_cls', [nn.ReLU, nn.GELU])
sparsity_parametrization = pytest.mark.parametrize('sparsity', [0.1, 0.25, 0.5, 0.75, 0.9])
debug_info_parametrization = pytest.mark.parametrize('debug_info', [True, False])


# ============================================================================
# TESTS - Decorator functionality
# ============================================================================

class TestAnalyticalModuleDecorator:
    """Tests for analytical_module decorator."""

    @activation_parametrization
    @debug_info_parametrization
    def test_creates_new_class_with_debug_attributes(self, activation_cls, debug_info):
        """Should create new class with debug_info, in_activation, out_activation."""
        AnalyticalReLU = analytical_module(activation_cls)
        module = AnalyticalReLU(debug_info=debug_info)

        assert hasattr(module, 'debug_info')
        assert hasattr(module, 'in_activation')
        assert hasattr(module, 'out_activation')
        assert module.debug_info is debug_info

    def test_debug_info_disabled_does_not_store_activations(self, sample_input):
        """When debug_info=False, activations should not be stored."""
        AnalyticalReLU = analytical_module(nn.ReLU)
        module = AnalyticalReLU(debug_info=False)

        _ = module(sample_input)

        assert module.in_activation is None
        assert module.out_activation is None

    @activation_parametrization
    def test_debug_info_enabled_stores_activations(self, sample_input, activation_cls):
        """When debug_info=True, activations should be stored."""
        AnalyticalReLU = analytical_module(activation_cls)
        module = AnalyticalReLU(debug_info=True)

        y = module(sample_input)

        assert module.in_activation is not None
        assert module.out_activation is not None
        assert torch.equal(module.in_activation, sample_input)
        assert torch.equal(module.out_activation, y)

    @activation_parametrization
    def test_preserves_class_name_with_analytical_prefix(self, activation_cls):
        """Class name should include 'Analytical' prefix."""
        AnalyticalReLU = analytical_module(activation_cls)

        assert AnalyticalReLU.__name__.startswith('Analytical')
        assert AnalyticalReLU.__name__.endswith(activation_cls.__name__)


class TestTopKSparseModuleDecorator:
    """Tests for topk_sparse_module decorator."""

    @activation_parametrization
    @sparsity_parametrization
    def test_creates_new_class_with_sparsity_attributes(self, activation_cls, sparsity):
        """Should create new class with sparsity_level and post_sparsity."""
        SparseReLU = topk_sparse_module(activation_cls)
        module = SparseReLU(sparsity_level=sparsity, post_sparsity=True)

        assert hasattr(module, 'sparsity_level')
        assert hasattr(module, 'post_sparsity')
        assert module.sparsity_level == sparsity
        assert module.post_sparsity is True

    @pytest.mark.parametrize('invalid_sparsity', [-0.5, 1.5, 2.0])
    def test_rejects_invalid_sparsity_levels(self, invalid_sparsity):
        """Should reject sparsity levels outside [0, 1] range."""
        SparseReLU = topk_sparse_module(nn.ReLU)

        with pytest.raises(AssertionError):
            SparseReLU(sparsity_level=invalid_sparsity)

    @activation_parametrization
    def test_no_sparsity_when_level_is_none(self, sample_input, activation_cls):
        """When sparsity_level=None, output should equal standard module."""
        SparseReLU = topk_sparse_module(activation_cls)
        sparse_module = SparseReLU(sparsity_level=None)
        normal_module = activation_cls()

        y_sparse = sparse_module(sample_input)
        y_normal = normal_module(sample_input)

        assert torch.allclose(y_sparse, y_normal)

    @activation_parametrization
    def test_full_sparsity_zeros_out_all_activations(self, sample_input, activation_cls):
        """When sparsity_level=1.0, all activations should be zero."""
        SparseReLU = topk_sparse_module(activation_cls)
        sparse_module = SparseReLU(sparsity_level=1.0, post_sparsity=True)

        y_sparse = sparse_module(sample_input)
        y_normal = torch.zeros_like(sample_input)

        assert torch.allclose(y_sparse, y_normal)
        
    def test_full_sparsity_zeros_out_all_activations_sigmoid(self, sample_input):
        """When sparsity_level=1.0, all activations should be zero."""
        SparseReLU = topk_sparse_module(nn.Sigmoid)
        sparse_module = SparseReLU(sparsity_level=1.0, post_sparsity=True)

        y_sparse = sparse_module(sample_input)
        y_normal = torch.zeros_like(sample_input)

        assert torch.allclose(y_sparse, y_normal)
    
        SparseReLU = topk_sparse_module(nn.Sigmoid)
        sparse_module = SparseReLU(sparsity_level=1.0, post_sparsity=False)

        y_sparse = sparse_module(sample_input)
        y_normal = torch.full_like(sample_input, 0.5)

        assert torch.allclose(y_sparse, y_normal)

    @activation_parametrization
    @sparsity_parametrization
    def test_sparsity_zeros_out_small_activations(self, batch_input, activation_cls, sparsity):
        """Should zero out approximately correct percentage of small values."""
        SparseReLU = topk_sparse_module(activation_cls)
        module = SparseReLU(sparsity_level=sparsity, post_sparsity=True)

        y = module(batch_input)

        zero_count = (y == 0).sum().item()
        total_count = y.numel()
        sparsity_ratio = zero_count / total_count

        # Allow 10% tolerance due to discrete top-k behavior
        if activation_cls != nn.ReLU:
            assert abs(sparsity_ratio - sparsity) < 0.1
        else:
            # For ReLU, we expect the sparsity to be higher due to zeroing out negative values
            assert sparsity_ratio >= sparsity - 0.1

    @activation_parametrization
    def test_preserves_class_name_with_topk_prefix(self, activation_cls):
        """Class name should include 'TopK' prefix."""
        SparseReLU = topk_sparse_module(activation_cls)

        assert SparseReLU.__name__.startswith('TopK')
        assert SparseReLU.__name__.endswith(activation_cls.__name__)

class TestCombinedDecorators:
    """Tests for using both analytical_module and topk_sparse_module together."""

    @activation_parametrization
    @sparsity_parametrization
    @debug_info_parametrization
    def test_combined_decorators_work_together(self, batch_input, activation_cls, sparsity, debug_info):
        """Should apply both analytical and sparsity modifications correctly."""
        AnalyticalSparseReLU = analytical_module(topk_sparse_module(activation_cls))
        module = AnalyticalSparseReLU(sparsity_level=sparsity, post_sparsity=True, debug_info=debug_info)

        y = module(batch_input.clone())

        # Check debug info attributes
        if debug_info:
            assert module.in_activation is not None
            assert module.out_activation is not None
            assert torch.equal(module.in_activation, batch_input)
            assert torch.equal(module.out_activation, y)
        else:
            assert module.in_activation is None
            assert module.out_activation is None

        # Check sparsity level approximately correct
        zero_count = (y == 0).sum().item()
        total_count = y.numel()
        sparsity_ratio = zero_count / total_count

        # For ReLU, we expect the sparsity to be higher due to zeroing out negative values
        if activation_cls != nn.ReLU:
            assert abs(sparsity_ratio - sparsity) < 0.1
        else:
            assert sparsity_ratio >= sparsity - 0.1

    
    @activation_parametrization
    @sparsity_parametrization
    @debug_info_parametrization
    def test_inner_application_order(self, batch_input, activation_cls, sparsity, debug_info):
        """Should apply decorators in correct order (analytical outside topk)."""
        AnalyticalSparseReLU = topk_sparse_module(analytical_module(activation_cls))
        module = AnalyticalSparseReLU(sparsity_level=sparsity, post_sparsity=True, debug_info=debug_info)

        y = module(batch_input.clone())

        # Check debug info attributes
        if debug_info:
            assert module.in_activation is not None
            assert module.out_activation is not None
            assert torch.equal(module.in_activation, batch_input)
            # assert not torch.equal(module.out_activation, y) # TODO: Fix this check to work with low sparsity levels
        else:
            assert module.in_activation is None
            assert module.out_activation is None

        # Check sparsity level approximately correct
        zero_count = (y == 0).sum().item()
        total_count = y.numel()
        sparsity_ratio = zero_count / total_count

        # For ReLU, we expect the sparsity to be higher due to zeroing out negative values
        if activation_cls != nn.ReLU:
            assert abs(sparsity_ratio - sparsity) < 0.1
        else:
            assert sparsity_ratio >= sparsity - 0.1
        
        AnalyticalSparseReLU = topk_sparse_module(analytical_module(activation_cls))
        module = AnalyticalSparseReLU(sparsity_level=sparsity, post_sparsity=False, debug_info=debug_info)

        y = module(batch_input.clone())

        # Check debug info attributes
        if debug_info:
            assert module.in_activation is not None
            assert module.out_activation is not None
            assert not torch.equal(module.in_activation, batch_input)
            assert torch.equal(module.out_activation, y)
        else:
            assert module.in_activation is None
            assert module.out_activation is None

        # Check sparsity level approximately correct
        zero_count = (y == 0).sum().item()
        total_count = y.numel()
        sparsity_ratio = zero_count / total_count

        # For ReLU, we expect the sparsity to be higher due to zeroing out negative values
        if activation_cls != nn.ReLU:
            assert abs(sparsity_ratio - sparsity) < 0.1
        else:
            assert sparsity_ratio >= sparsity - 0.1
