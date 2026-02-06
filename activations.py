import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ReLU
from torch.autograd import Function
# relu squared with optional signing

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class NoisyReLU(nn.Module):
    def __init__(self, alpha=1.0, c=1.0, noise_type='half-normal'):
        super().__init__()
        self.alpha = alpha
        self.c = c
        self.noise_type = noise_type
        self.p = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        if not self.training:
            # Pure ReLU at test time - maintains sparsity
            return F.relu(x)
        
        # Training time with noise
        h_x = F.relu(x)
        u_x = x
        delta = torch.where(x < 0, -x, torch.zeros_like(x))
        
        sigma = self.c * torch.square(torch.sigmoid(self.p * delta) - 0.5)
        direction = torch.where(x < 0, torch.ones_like(x), torch.zeros_like(x))
        
        if self.noise_type == 'half-normal':
            epsilon = torch.abs(torch.randn_like(x))
            noise = direction * sigma * epsilon
        else:
            epsilon = torch.randn_like(x)
            noise = direction * sigma * epsilon
        
        return self.alpha * h_x + (1 - self.alpha) * u_x + noise

ALPHA = 1.67

def b_silu_forward(x):
    sigma_x = torch.sigmoid(x)
    return (x + ALPHA) * sigma_x - ALPHA / 2.0

def b_silu_backward(x):
    # Derivative of B-SiLU: sigma(x) + (x + alpha) * sigma(x) * (1 - sigma(x))
    sigma_x = torch.sigmoid(x)
    return sigma_x + (x + ALPHA) * sigma_x * (1.0 - sigma_x)

# Define the custom autograd function for SUGAR (Surrogate Gradient for ReLU)
class SUGARBSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input 'x' for the backward pass
        ctx.save_for_backward(x)
        # Forward pass is simply ReLU(x)
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass uses the B-SiLU derivative as the surrogate gradient
        x, = ctx.saved_tensors
        # Calculate the B-SiLU derivative at input 'x'
        b_silu_grad = b_silu_backward(x)
        # Multiply with the incoming gradient
        return grad_output * b_silu_grad

# Define the nn.Module wrapper
class SUGARBSiLU(nn.Module):
    def forward(self, x):
        return SUGARBSiLUFunction.apply(x)
