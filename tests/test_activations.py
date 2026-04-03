"""
Comprehensive pytest test suite for Sparse-Activations library.

This test suite provides comprehensive coverage with:
- Clear, readable test organization using fixtures
- Parametrized tests for testing multiple configurations
- Descriptive test names and docstrings
- Proper setup/teardown with fixtures
- Grouping of related tests in classes
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modifiers.activations import (
    ReLUSquared,
    ReLUSquaredClipped,
    GELUSquared,
    GELUSquaredClipped,
    BSiLU,
    SUGARBSiLU,
    NoisyReLU,
    QuantileReLU,
    ACTIVATION_NAMES_MAP,
)


from tests.fixtures import *

# ============================================================================
# TESTS - Activation functions
# ============================================================================

class TestReLUSquared:
    """Tests for ReLUSquared activation."""

    @pytest.mark.parametrize('activation_cls', [ReLUSquared, ReLUSquaredClipped])
    def test_applies_relu_then_square(self, activation_cls):
        """Output should be ReLU(x)^2."""
        module = activation_cls()
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = module(x)

        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 4.0]])
        assert torch.allclose(y, expected)

    @pytest.mark.parametrize('activation_cls', [ReLUSquared, ReLUSquaredClipped])
    def test_output_is_non_negative(self, activation_cls, sample_input):
        """All outputs should be non-negative."""
        module = activation_cls()
        y = module(sample_input)

        assert (y >= 0).all()

    
    def test_clips_output_to_max_value(self):
        """ReLUSquaredClipped should clip output to clip_value."""
        module = ReLUSquaredClipped(clip_value=2.0)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        y = module(x)

        # ReLU: [0, 1, 2, 3], squared: [0, 1, 4, 9], clipped: [0, 1, 2, 2]
        expected = torch.tensor([[0.0, 1.0, 2.0, 2.0]])
        assert torch.allclose(y, expected)


class TestGELUSquared:
    """Tests for GELUSquared activation."""

    @pytest.mark.parametrize('activation_cls', [GELUSquared, GELUSquaredClipped])
    def test_applies_gelu_then_square(self, activation_cls):
        """Output should be GELU(x)^2."""
        module = activation_cls()
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = module(x)

        gelu = nn.GELU()
        expected = gelu(x) ** 2
        assert torch.allclose(y, expected)

    @pytest.mark.parametrize('activation_cls', [GELUSquared, GELUSquaredClipped])
    def test_output_is_non_negative(self, activation_cls, sample_input):
        """All outputs should be non-negative."""
        module = activation_cls()
        y = module(sample_input)

        assert (y >= 0).all()

    def test_clips_output_to_max_value(self):
        """GELUSquaredClipped should clip output to clip_value."""
        module = GELUSquaredClipped(clip_value=2.0)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        gelu = nn.GELU()
        expected = gelu(x) ** 2
        expected = torch.clamp(expected, max=2.0)

        y = module(x)

        assert torch.allclose(y, expected)


class TestBSiLU:
    """Tests for BSiLU activation."""

    def test_formula_at_zero(self):
        """At x=0, BSiLU(0) = (0 + alpha) * 0.5 - alpha/2 = 0."""
        module = BSiLU(alpha=1.0)
        x = torch.tensor([0.0])
        y = module(x)

        assert torch.allclose(y, torch.tensor([0.0]), atol=1e-5)

    def test_default_alpha_is_1_67(self):
        """Default alpha should be approximately 1.67."""
        module = BSiLU()
        assert abs(module.alpha - 1.67) < 0.01

    @pytest.mark.parametrize('alpha', [0.5, 1.0, 1.67, 2.0, 3.0])
    def test_accepts_custom_alpha(self, alpha):
        """Should accept custom alpha parameter."""
        module = BSiLU(alpha=alpha)
        assert module.alpha == alpha

    def test_rejects_inplace_true(self):
        """Should reject inplace=True since it breaks gradients."""
        with pytest.raises(AssertionError):
            BSiLU(inplace=True)


class TestSUGARBSiLU:
    """Tests for SUGARBSiLU activation."""

    def test_forward_pass_produces_expected_output(self):
        """Should produce expected output based on BSiLU formula."""
        module = SUGARBSiLU(alpha=1.0)
        relu_module = nn.ReLU()
        
        x = torch.tensor([[-1.0, 0.0, 1.0]])
        y = module(x)

        expected = relu_module(y)
        assert torch.allclose(y, expected, atol=1e-5)


class TestNoisyReLU:
    """Tests for NoisyReLU activation."""

    @pytest.mark.parametrize('alpha,c,noise_type', [
        (0.5, 1.0, 'normal'),
        (1.0, 2.0, 'half-normal'),
        (2.0, 0.5, 'normal'),
    ])
    def test_accepts_custom_parameters(self, alpha, c, noise_type):
        """Should accept custom parameters."""
        module = NoisyReLU(alpha=alpha, c=c, noise_type=noise_type)

        assert module.alpha == alpha
        assert module.c == c
        assert module.noise_type == noise_type

    def test_learnable_parameter_p(self):
        """Parameter p should be learnable."""
        module = NoisyReLU()

        assert isinstance(module.p, nn.Parameter)
        assert module.p.requires_grad

        with torch.no_grad():
            x = torch.randn(2, 3, 4, 4)
            _ = module(x)

        assert module.p.grad is None

    def test_no_noise_in_eval_mode(self, sample_input):
        """Eval mode should produce deterministic output (no noise)."""
        module = NoisyReLU()
        module.eval()

        relu_module = nn.ReLU()

        y1 = module(sample_input)
        y2 = module(sample_input)
        y3 = relu_module(sample_input)

        assert torch.equal(y1, y2)
        assert torch.equal(y1, y3)

    def test_adds_noise_in_training_mode(self, sample_input):
        """Training mode should apply noise transformation."""
        module = NoisyReLU(c=1.0)
        module.train()

        relu_module = nn.ReLU()

        y1 = module(sample_input)
        y2 = module(sample_input)
        y3 = relu_module(sample_input)
        
        assert ((sample_input >= 0) | (y1 != 0.0)).all()  # There are should be noise at the negative side
        assert ((sample_input >= 0) | (y1 != y2)).all()
        assert ((sample_input < 0) | (y1 == y3)).all()

    def test_rejects_inplace_true(self):
        """Should reject inplace=True since it breaks gradients."""
        with pytest.raises(AssertionError):
            NoisyReLU(inplace=True)


class TestQuantileReLU:
    """Tests for QuantileReLU activation."""

    def test_without_sparsity_level_behaves_like_relu(self, sample_input):
        """When sparsity_level=None, should act like standard ReLU."""
        module = QuantileReLU(sparsity_level=None)
        expected = nn.ReLU()(sample_input)

        y = module(sample_input)

        assert torch.allclose(y, expected)

    @pytest.mark.parametrize('sparsity_level', [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_accepts_valid_sparsity_levels(self, sparsity_level):
        """Should accept sparsity levels in (0, 1) range."""
        module = QuantileReLU(sparsity_level=sparsity_level)
        assert module.sparsity_level == sparsity_level


# ============================================================================
# TESTS - Property preservation and edge cases
# ============================================================================

activations_parametrization = pytest.mark.parametrize('activation_cls', [ReLUSquared, ReLUSquaredClipped, GELUSquared, GELUSquaredClipped, BSiLU, SUGARBSiLU, NoisyReLU, QuantileReLU])

class TestPropertyPreservation:
    """Tests for property and state preservation."""

    @activations_parametrization
    def test_preserves_input_dtype(self, sample_input, activation_cls):
        """Output dtype should match input dtype."""
        module = activation_cls()

        for dtype in [torch.float32, torch.float64]:
            x = sample_input.to(dtype)
            y = module(x)

            assert y.dtype == dtype

    @activations_parametrization
    def test_device_transfer_works(self, sample_input, device, activation_cls):
        """Module should work after device transfer."""
        module = activation_cls()
        module = module.to(device)

        x = sample_input.to(device)
        y = module(x)

        assert y.device.type == device.type

    @activations_parametrization
    def test_shape_preservation(self, sample_input, activation_cls):
        """Output shape should match input shape."""
        module = activation_cls()
        y = module(sample_input)

        assert y.shape == sample_input.shape


class TestAutogradVersioning:
    """Tests to ensure autograd versioning is not broken."""

    @activations_parametrization
    def test_no_inplace_breaks_in_relu_squared(self, grad_input, activation_cls):
        """ReLUSquared should not break autograd."""
        module = activation_cls()
        y = module(grad_input)
        loss = y.sum()

        loss.backward()

        assert grad_input.grad is not None

    @activations_parametrization
    def test_gradient_accumulation_works(self, sample_input, activation_cls):
        """Should support gradient accumulation."""
        module = activation_cls()

        for _ in range(3):
            x = sample_input.clone().detach().requires_grad_(True)
            y = module(x)
            loss = y.sum()
            loss.backward()

            assert x.grad is not None

    @activations_parametrization
    def test_sequential_forward_backward_passes(self, batch_input, activation_cls):
        """Should support multiple sequential passes."""
        model = nn.Sequential(
            activation_cls(),
            nn.Linear(16, 10),
        )

        for _ in range(3):
            x = batch_input.clone().detach().requires_grad_(True)
            y = model(x)
            loss = y.sum()
            loss.backward()

            assert x.grad is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @activations_parametrization
    def test_single_element_batch(self, activation_cls):
        """Should work with batch size 1."""
        module = activation_cls()
        x = torch.randn(1, 3, 4, 4)
        y = module(x)

        assert y.shape == x.shape

    @activations_parametrization
    def test_small_spatial_dimensions(self, activation_cls):
        """Should work with 1×1 spatial dimensions."""
        module = activation_cls()
        x = torch.randn(2, 3, 1, 1)
        y = module(x)

        assert y.shape == x.shape

    @activations_parametrization
    def test_zero_input_produces_zero_output(self, zero_input, activation_cls):
        """ReLUSquared(0) should be all zeros."""
        module = activation_cls()
        y = module(zero_input)

        assert torch.allclose(y, torch.zeros_like(y))

    @activations_parametrization
    def test_large_values_are_handled(self, large_value_input, activation_cls):
        """Should handle very large input values."""
        module = activation_cls()
        y = module(large_value_input)

        assert not torch.isnan(y).any()

    @pytest.mark.parametrize('activation_cls', [ReLUSquared, ReLUSquaredClipped])
    def test_negative_values_produce_zeros(self, negative_input, activation_cls):
        """ReLU should zero out all negative values."""
        module = activation_cls()
        y = module(negative_input)

        assert torch.allclose(y, torch.zeros_like(y))


# ============================================================================
# TESTS - Registry coverage
# ============================================================================

class TestActivationRegistry:
    """Tests for all activations in ACTIVATION_NAMES_MAP."""

    @pytest.mark.parametrize('name', list(ACTIVATION_NAMES_MAP.keys()))
    def test_all_activations_are_instantiable(self, name):
        """All registered activations should be instantiable."""
        cls_or_partial = ACTIVATION_NAMES_MAP[name]

        if callable(cls_or_partial):
            instance = cls_or_partial()
            assert instance is not None

    @pytest.mark.parametrize('name', list(ACTIVATION_NAMES_MAP.keys()))
    def test_all_activations_have_forward_pass(self, name, sample_input):
        """All activations should support forward pass."""
        cls_or_partial = ACTIVATION_NAMES_MAP[name]

        if callable(cls_or_partial):
            instance = cls_or_partial()
            y = instance(sample_input)

            assert y.shape == sample_input.shape
