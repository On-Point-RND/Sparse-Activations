import pytest
import torch
import torch.nn as nn


# ============================================================================
# FIXTURES - Setup and teardown helpers
# ============================================================================

@pytest.fixture
def sample_input():
    """Standard input tensor (2, 3, 4, 4)."""
    return torch.randn(2, 3, 4, 4)


@pytest.fixture
def small_input():
    """Small input tensor (1, 1, 4, 4)."""
    return torch.randn(1, 1, 4, 4)


@pytest.fixture
def batch_input():
    """Large batch input tensor (8, 64, 16, 16)."""
    return torch.randn(8, 64, 16, 16)


@pytest.fixture
def layer_norm_input():
    """Input suitable for LayerNorm (4, 100)."""
    return torch.randn(4, 100)


@pytest.fixture
def batch_norm_input():
    """Input suitable for BatchNorm2d (8, 32, 16, 16)."""
    return torch.randn(8, 32, 16, 16)


@pytest.fixture
def grad_input(sample_input):
    """Input tensor that requires gradients."""
    sample_input.requires_grad_(True)
    return sample_input


@pytest.fixture
def zero_input():
    """Zero-valued input tensor."""
    return torch.zeros(2, 3, 4, 4)


@pytest.fixture
def large_value_input():
    """Input with very large values (1e6)."""
    return torch.ones(2, 3, 4, 4) * 1e6


@pytest.fixture
def negative_input():
    """Input with negative values (-5)."""
    return torch.ones(2, 3, 4, 4) * -5.0


@pytest.fixture
def model_with_gelu():
    """Model with GELU activations."""
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.GELU(approximate='tanh'),
        nn.Linear(10, 10),
        nn.GELU(approximate='none')
    )


@pytest.fixture
def model_with_batchnorm():
    """Model with BatchNorm2d layers."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 3),
        nn.BatchNorm2d(128),
    )


@pytest.fixture
def device():
    """Detect and return available device (cuda or cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
